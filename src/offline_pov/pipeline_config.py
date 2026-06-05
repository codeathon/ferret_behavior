"""
Configuration for offline ferret POV pipeline runs.

External tool paths default to the Scholl lab layout but can be overridden
via ``configs/offline_pipeline.json`` so the same entrypoint works on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field


class ExternalToolPaths(BaseModel):
	"""Subprocess interpreters and scripts used by ``full_pipeline.py``."""

	skellyclicker_python: str = "/home/scholl-lab/anaconda3/envs/skellyclicker/bin/python"
	skellyclicker_script: str = "/home/scholl-lab/skellyclicker/skellyclicker/scripts/process_recording.py"
	triangulation_python: str = "/home/scholl-lab/Documents/git_repos/dlc_to_3d/.venv/bin/python"
	triangulation_script: str = (
		"/home/scholl-lab/Documents/git_repos/dlc_to_3d/dlc_reconstruction/dlc_to_3d.py"
	)
	calibration_python: str = "/home/scholl-lab/anaconda3/envs/fmc/bin/python"
	calibration_script: str = (
		"/home/scholl-lab/Documents/git_repos/freemocap/experimental/batch_process/"
		"headless_calibration.py"
	)


class OfflinePipelineConfig(BaseModel):
	"""
	Runtime settings for ``run_offline_pov``.

	``session_root`` may be the session folder or ``full_recording`` path.
	"""

	session_root: Path | None = None
	clip_name: str | None = None
	calibration_toml_path: Path | None = None
	external_tools: ExternalToolPaths = Field(default_factory=ExternalToolPaths)
	local_inspection_dir: Path = Field(
		default_factory=lambda: Path.home() / "ferret_pov_inspection"
	)
	include_eye: bool = True
	overwrite_synchronization: bool = False
	overwrite_calibration: bool = False
	overwrite_dlc: bool = False
	overwrite_triangulation: bool = False
	overwrite_eye_postprocessing: bool = False
	overwrite_skull_postprocessing: bool = False
	overwrite_gaze: bool = False
	reprocess_gaze_clip: bool = False

	@classmethod
	def from_json_file(cls, path: Path) -> Self:
		"""Load config from JSON; relative paths stay relative to the config file parent."""
		raw = json.loads(path.read_text(encoding="utf-8"))
		config = cls.model_validate(raw)
		base = path.parent.resolve()
		if config.session_root is not None:
			config.session_root = _resolve_optional_path(config.session_root, base)
		if config.calibration_toml_path is not None:
			config.calibration_toml_path = _resolve_optional_path(
				config.calibration_toml_path, base
			)
		config.local_inspection_dir = Path(config.local_inspection_dir).expanduser()
		if not config.local_inspection_dir.is_absolute():
			config.local_inspection_dir = (base / config.local_inspection_dir).resolve()
		return config


def _resolve_optional_path(value: Path, base: Path) -> Path:
	if value.is_absolute():
		return value
	return (base / value).resolve()


def load_offline_pipeline_config(path: Path | None = None) -> OfflinePipelineConfig:
	"""Load config from disk or return defaults."""
	if path is None:
		return OfflinePipelineConfig()
	return OfflinePipelineConfig.from_json_file(path)


def default_config_path() -> Path:
	"""Checked-in template config location."""
	return Path(__file__).resolve().parents[2] / "configs" / "offline_pipeline.json"
