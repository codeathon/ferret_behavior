"""
Arena geometry and per-wall video-crop settings for virtual rendering.

Wall ROIs are defined once per session in JSON; the video-crop pipeline reads
them to extract per-frame textures from overhead Basler streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field


class ArenaBoxMm(BaseModel):
	"""Axis-aligned arena box in millimeters (matches Rerun groundplane scale)."""

	center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 500.0])
	half_size: list[float] = Field(default_factory=lambda: [500.0, 500.0, 500.0])


class WallRoiPx(BaseModel):
	"""Pixel crop in a source overhead frame (x, y, width, height)."""

	x: int
	y: int
	w: int
	h: int


class WallScreenConfig(BaseModel):
	"""One wall screen: 3D placement plus overhead video crop source."""

	id: str
	# Screen quad center in arena mm; normal points outward from arena interior.
	screen_center_mm: list[float]
	screen_half_size_mm: list[float]
	normal: list[float]
	# Basler serial/name substring used to match synchronized mp4 stems.
	source_camera_name: str
	roi_px: WallRoiPx
	# Optional four image corners for perspective unwarp; order TL, TR, BR, BL.
	corner_px: list[list[int]] | None = None
	texture_width_px: int = 512
	texture_height_px: int = 512


class ArenaRenderGeometry(BaseModel):
	"""Full arena description consumed by texture extraction and Rerun viewer."""

	arena_mm: ArenaBoxMm = Field(default_factory=ArenaBoxMm)
	walls: list[WallScreenConfig] = Field(default_factory=list)
	stimulus_source: str = "video_crop"
	notes: str = ""

	@classmethod
	def from_json_file(cls, path: Path) -> Self:
		raw = json.loads(path.read_text(encoding="utf-8"))
		return cls.model_validate(raw)

	def write_json(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(
			json.dumps(self.model_dump(), indent="\t") + "\n",
			encoding="utf-8",
		)


def default_geometry_template_path() -> Path:
	return Path(__file__).resolve().parents[2] / "configs" / "arena_geometry.template.json"
