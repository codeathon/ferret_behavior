"""
Export a per-frame stimulus timeline for Unreal/Blender arena scenes.

References extracted wall texture paths keyed by frame index; no ferret pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.arena_render.arena_config import ArenaRenderGeometry
from src.arena_render.session_inputs import ArenaSessionInputs


def _relative_path(path: Path, base: Path) -> str:
	try:
		return str(path.resolve().relative_to(base.resolve()))
	except ValueError:
		return str(path.resolve())


def build_stimulus_timeline(
	inputs: ArenaSessionInputs,
	geometry: ArenaRenderGeometry,
	texture_root: Path,
	wall_texture_paths: dict[str, list[Path]],
	*,
	frame_stride: int = 1,
) -> dict:
	"""Build JSON-serializable timeline from extracted texture file lists."""
	frame_count = min(len(paths) for paths in wall_texture_paths.values()) if wall_texture_paths else 0
	timestamps_path = inputs.overhead_videos[0].timestamps_utc_path
	timestamps = np.load(timestamps_path, allow_pickle=True)
	frames = []
	for frame_idx in range(frame_count):
		actual_idx = frame_idx * frame_stride
		wall_textures = {
			wall_id: _relative_path(paths[frame_idx], texture_root)
			for wall_id, paths in wall_texture_paths.items()
		}
		ts_ns = int(timestamps[actual_idx]) if actual_idx < len(timestamps) else None
		frames.append(
			{
				"frame_index": actual_idx,
				"timestamp_utc_ns": ts_ns,
				"wall_textures": wall_textures,
			}
		)
	return {
		"session_root": str(inputs.session_root),
		"calibration_toml": str(inputs.calibration_toml),
		"sync_video_dir": str(inputs.sync_video_dir),
		"stimulus_source": geometry.stimulus_source,
		"frame_stride": frame_stride,
		"frame_count": frame_count,
		"walls": [wall.id for wall in geometry.walls],
		"frames": frames,
	}


def write_stimulus_timeline(
	timeline: dict,
	output_path: Path,
) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(timeline, indent="\t") + "\n", encoding="utf-8")
	return output_path
