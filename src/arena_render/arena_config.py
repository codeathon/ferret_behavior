"""
Arena geometry and per-wall video-crop settings for virtual rendering.

Wall ROIs are defined once per session in JSON; the video-crop pipeline reads
them to extract per-frame textures from overhead Basler streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


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


# Default 3D screen placement per wall id (mm, arena frame). Used when only roi_px is set.
_DEFAULT_WALL_3D: dict[str, dict[str, list[float]]] = {
	"north": {
		"screen_center_mm": [0.0, 500.0, 800.0],
		"screen_half_size_mm": [200.0, 150.0],
		"normal": [0.0, 1.0, 0.0],
	},
	"south": {
		"screen_center_mm": [0.0, -500.0, 800.0],
		"screen_half_size_mm": [200.0, 150.0],
		"normal": [0.0, -1.0, 0.0],
	},
	"east": {
		"screen_center_mm": [500.0, 0.0, 800.0],
		"screen_half_size_mm": [200.0, 150.0],
		"normal": [1.0, 0.0, 0.0],
	},
	"west": {
		"screen_center_mm": [-500.0, 0.0, 800.0],
		"screen_half_size_mm": [200.0, 150.0],
		"normal": [-1.0, 0.0, 0.0],
	},
}


class WallScreenConfig(BaseModel):
	"""One wall screen: 3D placement plus overhead video crop source."""

	id: str
	# 3D placement optional for texture extraction; defaults from wall id for Rerun/Unreal.
	screen_center_mm: list[float] | None = None
	screen_half_size_mm: list[float] | None = None
	normal: list[float] | None = None
	# Basler serial/name substring used to match synchronized mp4 stems.
	source_camera_name: str
	roi_px: WallRoiPx
	# Optional four image corners for perspective unwarp; order TL, TR, BR, BL.
	corner_px: list[list[int]] | None = None
	texture_width_px: int = 512
	texture_height_px: int = 512

	@model_validator(mode="after")
	def _apply_default_3d_placement(self) -> WallScreenConfig:
		"""Fill 3D fields from wall id when pick_wall_roi only wrote roi_px."""
		defaults = _DEFAULT_WALL_3D.get(self.id.lower(), _DEFAULT_WALL_3D["north"])
		if self.screen_center_mm is None:
			self.screen_center_mm = list(defaults["screen_center_mm"])
		if self.screen_half_size_mm is None:
			self.screen_half_size_mm = list(defaults["screen_half_size_mm"])
		if self.normal is None:
			self.normal = list(defaults["normal"])
		return self


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
