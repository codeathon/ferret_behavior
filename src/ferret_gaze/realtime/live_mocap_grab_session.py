"""
Basler grab session wired to live mocap inference + publish.

Uses :class:`~src.cameras.multicamera_recording.MultiCameraRecording` with
``frameset_sink`` set to a :class:`~src.ferret_gaze.realtime.grab_live_wiring.LiveMocapGrabPublishWire`
so combiner frame-sets are queued and processed on a background publisher thread.
"""

from __future__ import annotations

from pathlib import Path

from src.cameras.camera_config import configure_all_cameras
from src.cameras.multicamera_recording import MultiCameraRecording
from src.ferret_gaze.realtime.grab_live_wiring import LiveMocapGrabPublishWire
from src.ferret_gaze.realtime.latency_metrics import LatencySummary
from src.ferret_gaze.realtime.per_frame_compute import (
	RealtimeGazeFuser,
	RealtimeInferenceRuntime,
	RealtimeTriangulator,
	RollingEyeCalibrator,
)
from src.ferret_gaze.realtime.publisher import RealtimePublisher
from src.ferret_gaze.realtime.solver_benchmark import RealtimeSkullSolver
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def run_live_mocap_grab_n_frames_publish(
	*,
	output_path: Path,
	nir_only: bool,
	fps: float,
	binning_factor: int,
	hardware_triggering: bool,
	n_frames: int,
	publisher: RealtimePublisher,
	inference_runtime: RealtimeInferenceRuntime,
	triangulator: RealtimeTriangulator,
	stale_threshold_ms: float,
	wire_queue_size: int = 32,
	pace_hz: float | None = None,
	camera_exposure_overrides: dict[str, tuple[int, float]] | None = None,
	skull_solver: RealtimeSkullSolver | None = None,
	calibrator: RollingEyeCalibrator | None = None,
	fuser: RealtimeGazeFuser | None = None,
) -> LatencySummary | None:
	"""
	Open cameras, start the live publish consumer, grab ``n_frames`` synchronized sets, then stop.

	Does **not** close ``publisher``; the caller should do that after this returns.
	``n_frames`` must be positive.
	Optional ``calibrator`` / ``fuser`` override the stub defaults (see ``create_*`` factories).
	"""
	if n_frames < 1:
		raise ValueError("n_frames must be at least 1")

	mcr = MultiCameraRecording(output_path=output_path, nir_only=nir_only, fps=fps)
	mcr.open_camera_array()
	summary: LatencySummary | None = None
	try:
		mcr.set_max_num_buffer(240)
		mcr.set_fps(fps)
		mcr.set_image_resolution(binning_factor=binning_factor)
		mcr.set_hardware_triggering(hardware_triggering=hardware_triggering)
		configure_all_cameras(
			camera_array=mcr.camera_array,
			devices=mcr.devices,
			overrides=camera_exposure_overrides,
		)
		mcr.camera_information()
		mcr.create_video_writers_ffmpeg()

		wire = LiveMocapGrabPublishWire(max_queue_size=wire_queue_size)
		wire.start_background_publisher(
			publisher=publisher,
			inference_runtime=inference_runtime,
			triangulator=triangulator,
			stale_threshold_ms=stale_threshold_ms,
			pace_hz=pace_hz,
			skull_solver=skull_solver,
			calibrator=calibrator,
			fuser=fuser,
		)
		try:
			logger.info(
				"Live mocap grab: recording to %s, n_frames=%d, fps=%.2f, nir_only=%s",
				mcr.output_path,
				n_frames,
				fps,
				nir_only,
			)
			mcr.grab_n_frames(n_frames, frameset_sink=wire)
		finally:
			summary = wire.stop_background_publisher()
	finally:
		mcr.close_camera_array()
	if summary is not None:
		logger.info(
			"Live mocap grab session finished: latency packets=%d",
			summary.packet_count,
		)
	return summary
