"""
Extract per-frame wall textures from synchronized overhead video crops.

MVP uses rectangular ROI crops; optional four-corner homography unwarp rectifies
perspective when ``corner_px`` is set in arena geometry config.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.arena_render.arena_config import ArenaRenderGeometry, WallScreenConfig
from src.arena_render.session_inputs import ArenaSessionInputs, find_video_by_camera_name
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _crop_roi(frame: np.ndarray, wall: WallScreenConfig) -> np.ndarray:
	roi = wall.roi_px
	return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w].copy()


def _warp_to_texture(crop: np.ndarray, wall: WallScreenConfig) -> np.ndarray:
	if wall.corner_px is None or len(wall.corner_px) != 4:
		return cv2.resize(crop, (wall.texture_width_px, wall.texture_height_px))
	src = np.array(wall.corner_px, dtype=np.float32)
	dst = np.array(
		[
			[0, 0],
			[wall.texture_width_px - 1, 0],
			[wall.texture_width_px - 1, wall.texture_height_px - 1],
			[0, wall.texture_height_px - 1],
		],
		dtype=np.float32,
	)
	h_matrix = cv2.getPerspectiveTransform(src, dst)
	return cv2.warpPerspective(
		crop,
		h_matrix,
		(wall.texture_width_px, wall.texture_height_px),
		flags=cv2.INTER_LINEAR,
	)


def extract_wall_texture_frame(
	frame: np.ndarray,
	wall: WallScreenConfig,
) -> np.ndarray:
	"""Crop (and optionally unwarp) one overhead frame into a wall texture."""
	crop = _crop_roi(frame, wall)
	return _warp_to_texture(crop, wall)


def _open_capture(video_path: Path) -> cv2.VideoCapture:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	return cap


def extract_wall_textures_for_session(
	inputs: ArenaSessionInputs,
	geometry: ArenaRenderGeometry,
	output_dir: Path,
	*,
	frame_stride: int = 1,
	max_frames: int | None = None,
	image_format: str = "jpg",
) -> dict[str, list[Path]]:
	"""
	Write per-wall texture sequences under ``output_dir/wall_<id>/``.

	Returns a map of wall id -> ordered texture paths.
	"""
	output_dir = output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	wall_paths: dict[str, list[Path]] = {}
	captures: dict[str, cv2.VideoCapture] = {}

	try:
		for wall in geometry.walls:
			video = find_video_by_camera_name(inputs.overhead_videos, wall.source_camera_name)
			captures[wall.id] = _open_capture(video.path)
			(output_dir / f"wall_{wall.id}").mkdir(parents=True, exist_ok=True)
			wall_paths[wall.id] = []

		frame_limit = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures.values())
		if max_frames is not None:
			frame_limit = min(frame_limit, max_frames)

		written = 0
		for frame_idx in range(0, frame_limit, frame_stride):
			for wall in geometry.walls:
				cap = captures[wall.id]
				ok, frame = cap.read()
				if not ok:
					logger.warning("Wall %s ended early at frame %d", wall.id, frame_idx)
					continue
				texture = extract_wall_texture_frame(frame, wall)
				out_path = output_dir / f"wall_{wall.id}" / f"{frame_idx:06d}.{image_format}"
				cv2.imwrite(str(out_path), texture)
				wall_paths[wall.id].append(out_path)
			written += 1
			if written % 100 == 0:
				logger.info("Extracted %d frames", written)
	finally:
		for cap in captures.values():
			cap.release()

	logger.info("Wall texture extraction complete: %s", output_dir)
	return wall_paths
