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

import numpy as np

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
)
from src.ferret_gaze.realtime.publisher import RealtimePublisher
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


def basler_frameset_to_live_mocap_frame_set(basler: BaslerFrameSet, seq: int) -> LiveMocapFrameSet | None:
	"""
	Map a combiner :class:`BaslerFrameSet` to :class:`LiveMocapFrameSet`.

	Skips cameras whose ``payload`` is missing or not ``HxWx3`` uint8 BGR.
	Returns ``None`` if no valid images were collected.
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
	return LiveMocapFrameSet(
		seq=seq,
		anchor_utc_ns=basler.anchor_utc_ns,
		images_bgr=images_bgr,
	)


def build_synthetic_live_mocap_frame_sets(
	n_packets: int,
	*,
	n_cams: int = 2,
	height: int = 64,
	width: int = 64,
	anchor_step_ns: int = 1_000_000,
	anchor_base_ns: int | None = None,
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
		out.append(
			LiveMocapFrameSet(
				seq=seq,
				anchor_utc_ns=base + seq * anchor_step_ns,
				images_bgr=images,
			)
		)
	return out


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
			packet = gaze_packet_from_live_mocap_frame_set(fs)
			inference = inference_runtime.infer(packet, frame_set=fs)
			triangulated = triangulator.triangulate(inference)
			calibration = calibrator.update(triangulated)
			fused = fuser.fuse(packet, triangulated, calibration, inference)
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
