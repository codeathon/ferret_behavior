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
	triangulation_backend: str = Field(default="keypoint_centroid")
	# Optional explicit path to *camera_calibration.toml (defaults to session discovery in live path).
	calibration_toml_path: str | None = Field(default=None)

	acceptance_max_p95_ms: float = Field(default=80.0, gt=0.0)


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
