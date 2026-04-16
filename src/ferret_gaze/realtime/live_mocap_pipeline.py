"""
Live mocap orchestration: frame bundles -> inference -> triangulation -> publish.

Wires :class:`~src.ferret_gaze.realtime.live_frame_set.LiveMocapFrameSet` through
``infer(..., frame_set=...)`` and publishes fused gaze packets. Optional
conversion from :class:`~src.cameras.synchronization.realtime_sync.BaslerFrameSet`
for GrabLoopRunner sinks.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.cameras.synchronization.pupil_dual_eye_rings import PupilDualEyeRings
from src.cameras.synchronization.realtime_sync import BaslerFrameSet
from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.latency_metrics import (
	LatencySummary,
	RealtimeLatencyMetrics,
	format_latency_summary,
)
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.per_frame_compute import (
	RealtimeGazeFuser,
	RealtimeInferenceRuntime,
	RealtimeTriangulator,
	RollingEyeCalibrator,
	StubGazeFuser,
	StubRollingEyeCalibrator,
	apply_inference_to_gaze_packet,
)
from src.ferret_gaze.realtime.publisher import RealtimePublisher
from src.ferret_gaze.realtime.solver_benchmark import RealtimeSkullSolver
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def gaze_packet_from_live_mocap_frame_set(frame_set: LiveMocapFrameSet) -> RealtimeGazePacket:
	"""Build a gaze packet for one live tick; pose fields match transport scaffold defaults."""
	yaw = 0.05 * math.sin(frame_set.seq * 0.1)
	process_start_ns = time.time_ns()
	return RealtimeGazePacket(
		seq=frame_set.seq,
		capture_utc_ns=frame_set.anchor_utc_ns,
		process_start_ns=process_start_ns,
		publish_utc_ns=None,
		skull_position_xyz=(0.0, 0.0, 0.0),
		skull_quaternion_wxyz=(1.0, 0.0, yaw, 0.0),
		left_eye_origin_xyz=(-10.0, 25.0, 0.0),
		left_gaze_direction_xyz=(1.0, 0.0, 0.0),
		right_eye_origin_xyz=(10.0, 25.0, 0.0),
		right_gaze_direction_xyz=(1.0, 0.0, 0.0),
		confidence=1.0,
	)


def basler_frameset_to_live_mocap_frame_set(
	basler: BaslerFrameSet,
	seq: int,
	*,
	pupil_rings: PupilDualEyeRings | None = None,
	pupil_stale_max_delta_ns: int | None = None,
) -> LiveMocapFrameSet | None:
	"""
	Map a combiner :class:`BaslerFrameSet` to :class:`LiveMocapFrameSet`.

	Skips cameras whose ``payload`` is missing or not ``HxWx3`` uint8 BGR.
	Returns ``None`` if no valid images were collected.

	When ``pupil_rings`` is set, nearest wall-UTC-matched Pupil eye frames are attached
	(stale flags if buffer empty or delta exceeds ``pupil_stale_max_delta_ns``; default 80 ms).
	"""
	images_bgr: dict[int, np.ndarray] = {}
	for cam_id, frame in sorted(basler.frames_by_camera.items()):
		payload = frame.payload
		if payload is None or not isinstance(payload, np.ndarray):
			continue
		if payload.ndim == 3 and payload.shape[2] == 3:
			images_bgr[cam_id] = payload
	if not images_bgr:
		return None
	live = LiveMocapFrameSet(
		seq=seq,
		anchor_utc_ns=basler.anchor_utc_ns,
		images_bgr=images_bgr,
	)
	if pupil_rings is None:
		return live
	stale_ns = pupil_stale_max_delta_ns if pupil_stale_max_delta_ns is not None else 80_000_000
	a0, a1 = pupil_rings.associate(basler.anchor_utc_ns, stale_ns)
	return replace(
		live,
		eye0_bgr=a0.image,
		eye1_bgr=a1.image,
		pupil_eye0_utc_ns=a0.chosen_utc_ns,
		pupil_eye1_utc_ns=a1.chosen_utc_ns,
		pupil_eye0_delta_ns=a0.delta_ns,
		pupil_eye1_delta_ns=a1.delta_ns,
		pupil_eye0_stale=a0.stale,
		pupil_eye1_stale=a1.stale,
	)


def build_synthetic_live_mocap_frame_sets(
	n_packets: int,
	*,
	n_cams: int = 2,
	height: int = 64,
	width: int = 64,
	anchor_step_ns: int = 1_000_000,
	anchor_base_ns: int | None = None,
	attach_dummy_pupil_eyes: bool = False,
) -> list[LiveMocapFrameSet]:
	"""Produce trivial BGR bundles for dry-runs without cameras (tests / bench)."""
	if n_packets <= 0:
		raise ValueError("n_packets must be positive")
	if n_cams <= 0:
		raise ValueError("n_cams must be positive")
	base = anchor_base_ns if anchor_base_ns is not None else time.time_ns()
	out: list[LiveMocapFrameSet] = []
	for seq in range(n_packets):
		images = {i: np.zeros((height, width, 3), dtype=np.uint8) for i in range(n_cams)}
		anchor = base + seq * anchor_step_ns
		fs = LiveMocapFrameSet(seq=seq, anchor_utc_ns=anchor, images_bgr=images)
		if attach_dummy_pupil_eyes:
			# Tiny placeholders so the realtime path carries eye payloads like grab+Pupil rings.
			e0 = np.zeros((4, 4, 3), dtype=np.uint8)
			e1 = np.ones((4, 4, 3), dtype=np.uint8) * 40
			fs = replace(
				fs,
				eye0_bgr=e0,
				eye1_bgr=e1,
				pupil_eye0_utc_ns=anchor,
				pupil_eye1_utc_ns=anchor + 1,
				pupil_eye0_delta_ns=0,
				pupil_eye1_delta_ns=1,
				pupil_eye0_stale=False,
				pupil_eye1_stale=False,
			)
		out.append(fs)
	return out


def process_live_mocap_tick(
	frame_set: LiveMocapFrameSet,
	*,
	inference_runtime: RealtimeInferenceRuntime,
	triangulator: RealtimeTriangulator,
	calibrator: RollingEyeCalibrator,
	fuser: RealtimeGazeFuser,
	skull_solver: RealtimeSkullSolver | None = None,
) -> RealtimeGazePacket:
	"""
	Run infer -> triangulate -> calibrate -> fuse for one bundle (no publish).

	Optional ``skull_solver`` runs after triangulation (e.g. Kabsch orientation from
	keypoints). Inference may supply gaze vectors on :class:`~src.ferret_gaze.realtime.per_frame_compute.FrameInferenceResult`; those are copied onto the packet
	(via :func:`~src.ferret_gaze.realtime.per_frame_compute.apply_inference_to_gaze_packet`) before triangulation so calibrators and fusers see model-driven directions.
	The gaze fuser still receives triangulated skull position; rolling calibration
	runs after the optional skull solver so it sees the final quaternion.

	Used by the grab queue consumer and by :func:`run_live_mocap_compute_publish_session`.
	"""
	packet = gaze_packet_from_live_mocap_frame_set(frame_set)
	inference = inference_runtime.infer(packet, frame_set=frame_set)
	packet = apply_inference_to_gaze_packet(packet, inference)
	triangulated = triangulator.triangulate(inference)
	packet_for_fuse = packet.model_copy(update={"skull_position_xyz": triangulated.skull_position_xyz})
	if skull_solver is not None:
		packet_for_fuse = skull_solver.solve_with_context(
			packet_for_fuse,
			inference=inference,
			triangulated=triangulated,
		)
	calibration = calibrator.update(
		triangulated,
		inference=inference,
		packet=packet_for_fuse,
	)
	return fuser.fuse(packet_for_fuse, triangulated, calibration, inference)


def run_live_mocap_compute_publish_session(
	*,
	frame_sets: Sequence[LiveMocapFrameSet],
	publisher: RealtimePublisher,
	inference_runtime: RealtimeInferenceRuntime,
	triangulator: RealtimeTriangulator,
	hz: float,
	stale_threshold_ms: float,
	calibrator: RollingEyeCalibrator | None = None,
	fuser: RealtimeGazeFuser | None = None,
	skull_solver: RealtimeSkullSolver | None = None,
) -> LatencySummary:
	"""
	Run one live session: for each frame set, infer, triangulate, fuse, publish.

	Closes ``publisher`` in ``finally``. Respects ``hz`` with ``time.sleep`` between ticks.
	"""
	if hz <= 0:
		raise ValueError("hz must be positive")
	calibrator = calibrator or StubRollingEyeCalibrator()
	fuser = fuser or StubGazeFuser()
	metrics = RealtimeLatencyMetrics(stale_threshold_ms=stale_threshold_ms)
	period_s = 1.0 / hz
	confidence_sum = 0.0
	n_conf = 0
	logger.info(
		"Starting live mocap compute+publish: ticks=%d, hz=%.2f, stale_threshold_ms=%.1f",
		len(frame_sets),
		hz,
		stale_threshold_ms,
	)
	try:
		for fs in frame_sets:
			t_loop = time.perf_counter()
			fused = process_live_mocap_tick(
				fs,
				inference_runtime=inference_runtime,
				triangulator=triangulator,
				calibrator=calibrator,
				fuser=fuser,
				skull_solver=skull_solver,
			)
			fused = fused.model_copy(update={"publish_utc_ns": time.time_ns()})
			publisher.publish(fused)
			metrics.observe(packet=fused, now_utc_ns=time.time_ns())
			if fused.confidence is not None:
				confidence_sum += fused.confidence
				n_conf += 1
			elapsed = time.perf_counter() - t_loop
			sleep_s = period_s - elapsed
			if sleep_s > 0:
				time.sleep(sleep_s)
	finally:
		publisher.close()
	summary = metrics.summary()
	mean_conf = confidence_sum / n_conf if n_conf else 0.0
	logger.info(format_latency_summary(summary))
	logger.info("Live mocap session finished: mean_confidence=%.4f", mean_conf)
	return summary
