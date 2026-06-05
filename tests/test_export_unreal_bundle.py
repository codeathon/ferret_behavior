"""Tests for Unreal arena manifest export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import toml

from src.arena_render.export_unreal_bundle import build_unreal_arena_manifest, write_unreal_arena_manifest


def _write_calibration(cal_dir: Path) -> None:
	cal_dir.mkdir(parents=True, exist_ok=True)
	data = {
		"cam_0": {
			"name": "24676894",
			"world_position": [0.0, 0.0, 2000.0],
			"world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
			"matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
		}
	}
	with open(cal_dir / "session_test_camera_calibration.toml", "w", encoding="utf-8") as handle:
		toml.dump(data, handle)


def _build_extracted_session(tmp_path: Path) -> Path:
	session = tmp_path / "session_test"
	arena_render = session / "full_recording" / "arena_render"
	texture_root = arena_render / "wall_textures" / "wall_north"
	texture_root.mkdir(parents=True)
	(texture_root / "000000.jpg").write_bytes(b"\xff\xd8\xff")

	sync_dir = session / "full_recording" / "mocap_data" / "synchronized_corrected_videos"
	sync_dir.mkdir(parents=True)
	(sync_dir / "24676894_test.mp4").write_bytes(b"\x00")
	ts = np.array([0, int(1e9 / 90)], dtype=np.int64)
	np.save(sync_dir / "24676894_timestamps_utc.npy", ts)

	_write_calibration(session / "calibration")

	geometry = {
		"walls": [
			{
				"id": "north",
				"source_camera_name": "24676894",
				"roi_px": {"x": 0, "y": 0, "w": 10, "h": 10},
			}
		]
	}
	(session / "arena_geometry.json").write_text(json.dumps(geometry), encoding="utf-8")

	timeline = {
		"frame_count": 1,
		"walls": ["north"],
		"frames": [
			{
				"frame_index": 0,
				"timestamp_utc_ns": 0,
				"wall_textures": {"north": "wall_north/000000.jpg"},
			}
		],
	}
	(arena_render / "stimulus_timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
	return session


def test_build_unreal_arena_manifest(tmp_path: Path) -> None:
	session = _build_extracted_session(tmp_path)
	manifest = build_unreal_arena_manifest(session, session / "arena_geometry.json")
	assert manifest["frame_count"] == 1
	assert manifest["units"] == "cm"
	assert manifest["arena"]["half_size_cm"] == [50.0, 50.0, 50.0]
	assert len(manifest["cameras"]) == 1
	assert manifest["cameras"][0]["position_cm"][2] == 200.0


def test_write_unreal_arena_manifest(tmp_path: Path) -> None:
	session = _build_extracted_session(tmp_path)
	manifest = build_unreal_arena_manifest(session, session / "arena_geometry.json")
	out = write_unreal_arena_manifest(manifest, tmp_path / "manifest.json")
	assert out.is_file()
	loaded = json.loads(out.read_text(encoding="utf-8"))
	assert loaded["version"] == 1
