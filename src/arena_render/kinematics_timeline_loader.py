"""
Load skull and gaze kinematics from analyzable_output tidy CSVs for timeline merge.

Positions are converted mm -> cm for Unreal (1 uu = 1 cm). Gaze directions are unit vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

MM_TO_CM = 0.1


@dataclass(frozen=True)
class OfflinePoseArrays:
	"""Per-mocap-frame skull pose and binocular gaze in world coordinates."""

	frame_count: int
	skull_position_cm: np.ndarray  # (N, 3)
	skull_quaternion_wxyz: np.ndarray  # (N, 4)
	left_origin_cm: np.ndarray  # (N, 3)
	left_direction: np.ndarray  # (N, 3) unit
	right_origin_cm: np.ndarray  # (N, 3)
	right_direction: np.ndarray  # (N, 3) unit


def _resolve_analyzable_output(analyzable_output_dir: Path) -> Path:
	root = analyzable_output_dir.resolve()
	if (root / "skull_kinematics").is_dir():
		return root
	if (root / "analyzable_output" / "skull_kinematics").is_dir():
		return root / "analyzable_output"
	raise FileNotFoundError(
		f"No skull_kinematics/ under {analyzable_output_dir}. "
		"Expected data/session_.../analyzable_output/"
	)


def _extract_vector_by_frame(
	df: pl.DataFrame,
	trajectory: str,
	components: tuple[str, ...],
) -> np.ndarray:
	"""Build (num_frames, len(components)) array indexed by frame column."""
	sub = df.filter(pl.col("trajectory") == trajectory)
	if sub.is_empty():
		raise ValueError(f"Trajectory {trajectory!r} not found in kinematics CSV")
	max_frame = int(sub["frame"].max())
	out = np.zeros((max_frame + 1, len(components)), dtype=np.float64)
	for axis_idx, component in enumerate(components):
		axis_df = (
			sub.filter(pl.col("component") == component)
			.sort("frame")
			.select(["frame", "value"])
		)
		frames = axis_df["frame"].to_numpy()
		values = axis_df["value"].to_numpy()
		out[frames, axis_idx] = values
	return out


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms = np.where(norms > 1e-9, norms, 1.0)
	return vectors / norms


def load_offline_pose_arrays(analyzable_output_dir: Path) -> OfflinePoseArrays:
	"""Load skull + left/right gaze arrays from analyzable_output."""
	root = _resolve_analyzable_output(analyzable_output_dir)
	skull_csv = root / "skull_kinematics" / "skull_kinematics.csv"
	left_csv = root / "gaze_kinematics" / "left_gaze_kinematics.csv"
	right_csv = root / "gaze_kinematics" / "right_gaze_kinematics.csv"
	for path in (skull_csv, left_csv, right_csv):
		if not path.is_file():
			raise FileNotFoundError(f"Missing kinematics CSV: {path}")

	skull_df = pl.read_csv(skull_csv)
	left_df = pl.read_csv(left_csv)
	right_df = pl.read_csv(right_csv)

	skull_pos_mm = _extract_vector_by_frame(skull_df, "position", ("x", "y", "z"))
	skull_quat = _extract_vector_by_frame(skull_df, "orientation", ("w", "x", "y", "z"))

	left_origin_mm = _extract_vector_by_frame(left_df, "position", ("x", "y", "z"))
	left_target_mm = _extract_vector_by_frame(left_df, "keypoint__gaze_target", ("x", "y", "z"))
	right_origin_mm = _extract_vector_by_frame(right_df, "position", ("x", "y", "z"))
	right_target_mm = _extract_vector_by_frame(right_df, "keypoint__gaze_target", ("x", "y", "z"))

	left_dir = _normalize_rows(left_target_mm - left_origin_mm)
	right_dir = _normalize_rows(right_target_mm - right_origin_mm)

	frame_count = skull_pos_mm.shape[0]
	return OfflinePoseArrays(
		frame_count=frame_count,
		skull_position_cm=skull_pos_mm * MM_TO_CM,
		skull_quaternion_wxyz=skull_quat,
		left_origin_cm=left_origin_mm * MM_TO_CM,
		left_direction=left_dir,
		right_origin_cm=right_origin_mm * MM_TO_CM,
		right_direction=right_dir,
	)


def pose_dict_for_frame(pose: OfflinePoseArrays, frame_index: int) -> dict | None:
	"""Return JSON-serializable skull/gaze block for one mocap frame index."""
	if frame_index < 0 or frame_index >= pose.frame_count:
		return None
	return {
		"skull": {
			"position_cm": pose.skull_position_cm[frame_index].tolist(),
			"quaternion_wxyz": pose.skull_quaternion_wxyz[frame_index].tolist(),
		},
		"gaze": {
			"left": {
				"origin_cm": pose.left_origin_cm[frame_index].tolist(),
				"direction": pose.left_direction[frame_index].tolist(),
			},
			"right": {
				"origin_cm": pose.right_origin_cm[frame_index].tolist(),
				"direction": pose.right_direction[frame_index].tolist(),
			},
		},
	}
