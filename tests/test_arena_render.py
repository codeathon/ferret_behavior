"""Tests for arena-only virtual render helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import toml

from src.arena_render.arena_config import ArenaRenderGeometry
from src.arena_render.session_inputs import find_video_by_camera_name, list_overhead_videos
from src.arena_render.timeline_exporter import build_stimulus_timeline
from src.arena_render.validate_arena import validate_arena_session
from src.arena_render.wall_texture_extractor import extract_wall_texture_frame


def _write_minimal_calibration(cal_dir: Path) -> Path:
	cal_dir.mkdir(parents=True, exist_ok=True)
	data = {}
	for idx, name in enumerate(("24676894", "24908831", "cam2", "cam3", "cam4")):
		data[f"cam_{idx}"] = {
			"name": name,
			"world_position": [float(idx), 0.0, 2.0],
			"world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
			"matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
		}
	path = cal_dir / "session_test_camera_calibration.toml"
	with open(path, "w", encoding="utf-8") as handle:
		toml.dump(data, handle)
	return path


def _build_synced_session(tmp_path: Path) -> Path:
	session = tmp_path / "session_test"
	sync_dir = session / "full_recording" / "mocap_data" / "synchronized_corrected_videos"
	sync_dir.mkdir(parents=True)
	cal_path = _write_minimal_calibration(session / "calibration")

	for serial in ("24676894", "24908831"):
		video_path = sync_dir / f"{serial}_test.mp4"
		video_path.write_bytes(b"\x00")
		ts = np.array([1_000_000_000, 2_000_000_000], dtype=np.int64)
		np.save(sync_dir / f"{serial}_timestamps_utc.npy", ts)

	return session


def test_arena_geometry_roundtrip(tmp_path: Path) -> None:
	geometry = ArenaRenderGeometry.from_json_file(
		Path(__file__).resolve().parents[1] / "configs" / "arena_geometry.template.json"
	)
	out_path = tmp_path / "arena_geometry.json"
	geometry.write_json(out_path)
	loaded = ArenaRenderGeometry.from_json_file(out_path)
	assert loaded.stimulus_source == "video_crop"
	assert len(loaded.walls) == 4


def test_list_overhead_videos(tmp_path: Path) -> None:
	session = _build_synced_session(tmp_path)
	sync_dir = session / "full_recording" / "mocap_data" / "synchronized_corrected_videos"
	videos = list_overhead_videos(sync_dir)
	assert len(videos) == 2
	assert videos[0].frame_count == 2


def test_find_video_by_camera_name(tmp_path: Path) -> None:
	session = _build_synced_session(tmp_path)
	sync_dir = session / "full_recording" / "mocap_data" / "synchronized_corrected_videos"
	videos = list_overhead_videos(sync_dir)
	match = find_video_by_camera_name(videos, "24676894")
	assert "24676894" in match.stem


def test_validate_arena_session_passes(tmp_path: Path) -> None:
	session = _build_synced_session(tmp_path)
	report = validate_arena_session(session)
	assert report.ok


def test_extract_wall_texture_frame_resize() -> None:
	geometry = ArenaRenderGeometry.from_json_file(
		Path(__file__).resolve().parents[1] / "configs" / "arena_geometry.template.json"
	)
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	texture = extract_wall_texture_frame(frame, geometry.walls[0])
	assert texture.shape == (512, 512, 3)


def test_build_stimulus_timeline(tmp_path: Path) -> None:
	session = _build_synced_session(tmp_path)
	from src.arena_render.session_inputs import resolve_arena_session_inputs

	inputs = resolve_arena_session_inputs(session)
	geometry = ArenaRenderGeometry.from_json_file(
		Path(__file__).resolve().parents[1] / "configs" / "arena_geometry.template.json"
	)
	texture_root = tmp_path / "textures"
	wall_paths = {
		"north": [texture_root / "wall_north" / "000000.jpg"],
		"south": [texture_root / "wall_south" / "000000.jpg"],
	}
	timeline = build_stimulus_timeline(inputs, geometry, texture_root, wall_paths)
	assert timeline["frame_count"] == 1
	assert "north" in timeline["frames"][0]["wall_textures"]
