"""
Runtime configuration for realtime mode orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


class RealtimeRuntimeConfig(BaseModel):
	"""Centralized runtime knobs for Step 9 hardening."""

	transport_backend: str = Field(default="noop")
	transport_endpoint: str = Field(default="tcp://127.0.0.1:5556")
	transport_topic: str = Field(default="gaze.live")
	transport_payload_format: Literal["json", "msgpack"] = Field(default="msgpack")

	transport_packets: int = Field(default=120, ge=1)
	transport_hz: float = Field(default=60.0, gt=0.0)
	stale_threshold_ms: float = Field(default=80.0, gt=0.0)

	benchmark_packets: int = Field(default=120, ge=1)
	compute_packets: int = Field(default=120, ge=1)
	inference_backend: str = Field(default="stub")
	inference_model_path: str | None = Field(default=None)
	onnx_provider: str = Field(default="CPUExecutionProvider")
	# Used by ``onnx_images`` backend: resize each camera crop before NCHW feed.
	inference_images_height: int = Field(default=256, ge=1)
	inference_images_width: int = Field(default=256, ge=1)
	# If true, model outputs u,v in [0,1] relative to the resized input; else in resized pixel units.
	inference_output_uv_normalized: bool = Field(default=False)
	triangulation_backend: str = Field(default="keypoint_centroid")
	# Optional explicit path to *camera_calibration.toml (defaults to session discovery in live path).
	calibration_toml_path: str | None = Field(default=None)

	acceptance_max_p95_ms: float = Field(default=80.0, gt=0.0)

	# Orchestration: ``scaffold`` = synthetic transport + replay compute (legacy).
	# ``live_mocap`` = frame bundles -> infer -> triangulate -> publish (see live_mocap_pipeline).
	realtime_mode: Literal["scaffold", "live_mocap"] = Field(default="scaffold")
	# ``synthetic`` = offline bundles; ``grab`` = Basler MultiCameraRecording + frameset_sink wire.
	live_mocap_frame_source: Literal["synthetic", "grab"] = Field(default="synthetic")
	live_mocap_synthetic_camera_count: int = Field(default=2, ge=1)
	live_mocap_synthetic_height: int = Field(default=64, ge=1)
	live_mocap_synthetic_width: int = Field(default=64, ge=1)
	# --- grab-backed live_mocap (requires cameras + pylon) ---
	live_mocap_grab_output_path: str | None = Field(
		default=None,
		description="Directory for raw video output; defaults to run_pipeline recording_folder_path.",
	)
	live_mocap_grab_n_frames: int | None = Field(
		default=None,
		ge=1,
		description="Frames to grab; defaults to transport_packets when unset.",
	)
	live_mocap_grab_nir_only: bool = Field(default=False)
	live_mocap_grab_fps: float | None = Field(
		default=None,
		gt=0.0,
		description="Acquisition FPS; defaults to transport_hz when unset.",
	)
	live_mocap_grab_binning_factor: int = Field(default=2, ge=1, le=4)
	live_mocap_grab_hardware_trigger: bool = Field(default=True)
	live_mocap_grab_wire_queue_size: int = Field(default=32, ge=1)
	live_mocap_grab_pace_hz: float | None = Field(
		default=None,
		gt=0.0,
		description="Optional max publish rate for the consumer thread (None = no extra pacing).",
	)


def load_realtime_runtime_config(config_path: Path | None = None) -> RealtimeRuntimeConfig:
	"""
	Load realtime runtime config from JSON if provided; otherwise defaults.
	"""
	if config_path is None:
		return RealtimeRuntimeConfig()
	if not config_path.exists():
		logger.warning("Realtime config file not found: %s — using defaults", config_path)
		return RealtimeRuntimeConfig()
	with open(config_path, encoding="utf-8") as f:
		raw: Any = json.load(f)
	config = RealtimeRuntimeConfig.model_validate(raw)
	logger.info("Loaded realtime runtime config from %s", config_path)
	return config
