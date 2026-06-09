"""
Export a per-frame stimulus timeline for Unreal/Blender arena scenes.

References extracted wall texture paths keyed by frame index. Optional skull/gaze
pose blocks are merged from analyzable_output when requested.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.arena_render.arena_config import ArenaRenderGeometry
from src.arena_render.kinematics_timeline_loader import load_offline_pose_arrays, pose_dict_for_frame
from src.arena_render.session_inputs import ArenaSessionInputs
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


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


def merge_pose_into_timeline(
	timeline: dict,
	analyzable_output_dir: Path,
) -> dict:
	"""
	Attach per-frame skull pose and binocular gaze to an existing wall timeline.

	Uses each frame's ``frame_index`` to index mocap kinematics CSV rows.
	"""
	pose = load_offline_pose_arrays(analyzable_output_dir)
	merged_frames: list[dict] = []
	missing_pose = 0
	for frame in timeline.get("frames", []):
		frame_index = int(frame["frame_index"])
		merged = dict(frame)
		pose_block = pose_dict_for_frame(pose, frame_index)
		if pose_block is None:
			missing_pose += 1
		else:
			merged.update(pose_block)
		merged_frames.append(merged)

	if missing_pose:
		logger.warning(
			"%d timeline frames had no pose (mocap has %d frames)",
			missing_pose,
			pose.frame_count,
		)

	timeline = dict(timeline)
	timeline["frames"] = merged_frames
	timeline["has_pose"] = True
	timeline["pose_units"] = "cm"
	timeline["analyzable_output"] = str(analyzable_output_dir.resolve())
	timeline["mocap_frame_count"] = pose.frame_count
	logger.info(
		"Merged pose into %d timeline frames from %s",
		len(merged_frames),
		analyzable_output_dir,
	)
	return timeline


def write_stimulus_timeline(
	timeline: dict,
	output_path: Path,
) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(timeline, indent="\t") + "\n", encoding="utf-8")
	return output_path
