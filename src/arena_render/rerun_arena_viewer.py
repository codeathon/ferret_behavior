"""
Rerun viewer for arena geometry, camera frustums, and optional wall texture preview.

Arena-only scope: no ferret tracks or gaze overlays.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import rerun as rr
import toml

from src.arena_render.arena_config import ArenaRenderGeometry
from src.arena_render.session_inputs import ArenaSessionInputs
from src.rerun_viewer.rerun_utils.groundplane_and_origin import log_groundplane_and_origin
from src.rerun_viewer.rerun_utils.log_cameras import log_cameras


def log_arena_box(geometry: ArenaRenderGeometry, entity_path: str = "arena") -> None:
	"""Log configured arena box (overrides default 1 m cube when custom sizes are set)."""
	box = geometry.arena_mm
	rr.log(
		entity_path,
		rr.Boxes3D(
			centers=[box.center],
			half_sizes=[box.half_size],
			colors=[[220, 220, 220]],
		),
		static=True,
	)


def log_wall_planes(geometry: ArenaRenderGeometry, entity_path: str = "walls") -> None:
	"""Log wall screen centers as boxes for ROI placement debugging."""
	for wall in geometry.walls:
		half = [
			wall.screen_half_size_mm[0],
			wall.screen_half_size_mm[1],
			5.0,
		]
		rr.log(
			f"{entity_path}/{wall.id}",
			rr.Boxes3D(
				centers=[wall.screen_center_mm],
				half_sizes=[half],
				labels=[wall.id],
			),
			static=True,
		)


def log_wall_texture_preview(
	geometry: ArenaRenderGeometry,
	texture_root: Path,
	frame_index: int,
	entity_path: str = "wall_textures",
) -> None:
	"""Log one frame of extracted wall textures onto wall entities."""
	for wall in geometry.walls:
		image_path = texture_root / f"wall_{wall.id}" / f"{frame_index:06d}.jpg"
		if not image_path.is_file():
			image_path = texture_root / f"wall_{wall.id}" / f"{frame_index:06d}.png"
			if not image_path.is_file():
				continue
		rr.log(
			f"{entity_path}/{wall.id}",
			rr.Transform3D(translation=wall.screen_center_mm),
		)
		rr.log(
			f"{entity_path}/{wall.id}",
			rr.Pinhole(
				focal_length=wall.screen_half_size_mm[0],
				width=wall.texture_width_px,
				height=wall.texture_height_px,
			),
		)
		rr.log(f"{entity_path}/{wall.id}", rr.EncodedImage(path=image_path))


def launch_arena_rerun_viewer(
	inputs: ArenaSessionInputs,
	geometry: ArenaRenderGeometry | None = None,
	*,
	texture_root: Path | None = None,
	preview_frame: int | None = None,
	spawn: bool = True,
) -> str:
	"""Open Rerun with cameras, groundplane, and optional wall texture preview."""
	recording_name = f"arena_{inputs.session_root.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	rr.init(recording_name, spawn=spawn)
	calibration = toml.load(inputs.calibration_toml)
	log_cameras(calibration)
	log_groundplane_and_origin()
	if geometry is not None:
		log_arena_box(geometry)
		log_wall_planes(geometry)
		if texture_root is not None and preview_frame is not None:
			log_wall_texture_preview(geometry, texture_root, preview_frame)
	return recording_name
