"""Tests for offline skull/gaze timeline merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.arena_render.kinematics_timeline_loader import load_offline_pose_arrays, pose_dict_for_frame
from src.arena_render.timeline_exporter import merge_pose_into_timeline


def _write_tidy_csv(path: Path, rows: list[tuple]) -> None:
	lines = ["frame,timestamp_s,trajectory,component,value,units"]
	for frame, timestamp, trajectory, component, value, units in rows:
		lines.append(f"{frame},{timestamp},{trajectory},{component},{value},{units}")
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_analyzable_output(tmp_path: Path) -> Path:
	root = tmp_path / "analyzable_output"
	skull_dir = root / "skull_kinematics"
	gaze_dir = root / "gaze_kinematics"
	skull_dir.mkdir(parents=True)
	gaze_dir.mkdir(parents=True)

	skull_rows = []
	left_rows = []
	right_rows = []
	for frame in (0, 1):
		ts = float(frame)
		skull_rows.extend(
			[
				(frame, ts, "position", "x", 100.0 + frame, "mm"),
				(frame, ts, "position", "y", 200.0, "mm"),
				(frame, ts, "position", "z", 50.0, "mm"),
				(frame, ts, "orientation", "w", 1.0, "quaternion"),
				(frame, ts, "orientation", "x", 0.0, "quaternion"),
				(frame, ts, "orientation", "y", 0.0, "quaternion"),
				(frame, ts, "orientation", "z", 0.0, "quaternion"),
			]
		)
		left_rows.extend(
			[
				(frame, ts, "position", "x", 100.0 + frame, "mm"),
				(frame, ts, "position", "y", 200.0, "mm"),
				(frame, ts, "position", "z", 50.0, "mm"),
				(frame, ts, "keypoint__gaze_target", "x", 110.0 + frame, "mm"),
				(frame, ts, "keypoint__gaze_target", "y", 210.0, "mm"),
				(frame, ts, "keypoint__gaze_target", "z", 60.0, "mm"),
			]
		)
		right_rows.extend(
			[
				(frame, ts, "position", "x", 90.0 + frame, "mm"),
				(frame, ts, "position", "y", 190.0, "mm"),
				(frame, ts, "position", "z", 45.0, "mm"),
				(frame, ts, "keypoint__gaze_target", "x", 95.0 + frame, "mm"),
				(frame, ts, "keypoint__gaze_target", "y", 195.0, "mm"),
				(frame, ts, "keypoint__gaze_target", "z", 55.0, "mm"),
			]
		)

	_write_tidy_csv(skull_dir / "skull_kinematics.csv", skull_rows)
	_write_tidy_csv(gaze_dir / "left_gaze_kinematics.csv", left_rows)
	_write_tidy_csv(gaze_dir / "right_gaze_kinematics.csv", right_rows)
	return root


def test_load_offline_pose_arrays_mm_to_cm(tmp_path: Path) -> None:
	root = _build_analyzable_output(tmp_path)
	pose = load_offline_pose_arrays(root)
	assert pose.frame_count == 2
	assert pose.skull_position_cm[0, 0] == pytest.approx(10.0)
	assert pose.skull_position_cm[1, 0] == pytest.approx(10.1)


def test_merge_pose_into_timeline(tmp_path: Path) -> None:
	root = _build_analyzable_output(tmp_path)
	timeline = {
		"frame_count": 2,
		"frames": [
			{"frame_index": 0, "wall_textures": {"north": "wall_north/000000.jpg"}},
			{"frame_index": 1, "wall_textures": {"north": "wall_north/000001.jpg"}},
		],
	}
	merged = merge_pose_into_timeline(timeline, root)
	assert merged["has_pose"] is True
	assert merged["pose_units"] == "cm"
	assert "skull" in merged["frames"][0]
	assert merged["frames"][0]["skull"]["position_cm"][0] == 10.0
	assert "gaze" in merged["frames"][0]
	assert len(merged["frames"][0]["gaze"]["left"]["direction"]) == 3


def test_pose_dict_for_frame_out_of_range(tmp_path: Path) -> None:
	root = _build_analyzable_output(tmp_path)
	pose = load_offline_pose_arrays(root)
	assert pose_dict_for_frame(pose, 99) is None
