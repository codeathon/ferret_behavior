"""
Configuration for offline arena virtual rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field


class ArenaRenderConfig(BaseModel):
	"""Runtime settings for ``run_arena_render``."""

	session_root: Path | None = None
	calibration_toml_path: Path | None = None
	geometry_path: Path | None = None
	output_dir: Path | None = None
	frame_stride: int = 1
	max_frames: int | None = None

	@classmethod
	def from_json_file(cls, path: Path) -> Self:
		raw = json.loads(path.read_text(encoding="utf-8"))
		config = cls.model_validate(raw)
		base = path.parent.resolve()
		for field_name in ("session_root", "calibration_toml_path", "geometry_path", "output_dir"):
			value = getattr(config, field_name)
			if value is not None and not value.is_absolute():
				setattr(config, field_name, (base / value).resolve())
		return config


def default_config_path() -> Path:
	return Path(__file__).resolve().parents[2] / "configs" / "arena_render.json"


def load_arena_render_config(path: Path | None = None) -> ArenaRenderConfig:
	if path is None:
		return ArenaRenderConfig()
	return ArenaRenderConfig.from_json_file(path)
