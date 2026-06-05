"""Tests for offline POV session validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import toml

from src.offline_pov.validate_session import (
	ValidationLevel,
	resolve_session_paths,
	validate_session,
)
from src.utilities.folder_utilities.recording_folder import BaslerCamera


def _write_calibration_toml(path: Path, n_cams: int = 5) -> None:
	blocks = {}
	for i in range(n_cams):
		blocks[f"cam_{i}"] = {
			"name": f"cam{i}",
			"world_position": [float(i * 100), 0.0, 1500.0],
			"world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
			"matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
		}
	with open(path, "w", encoding="utf-8") as handle:
		toml.dump(blocks, handle)


def _build_minimal_raw_session(tmp_path: Path) -> Path:
	"""Session with raw videos + calibration TOML (pre-sync)."""
	session = tmp_path / "session_2025-01-01_ferret_420_test"
	full_recording = session / "full_recording"
	calibration = session / "calibration"

	mocap_raw = full_recording / "mocap_data" / "raw_videos"
	eye_videos = full_recording / "eye_data" / "eye_videos"
	mocap_raw.mkdir(parents=True)
	eye_videos.mkdir(parents=True)
	calibration.mkdir(parents=True)

	(mocap_raw / "timestamp_mapping.json").write_text(
		json.dumps({"starting_mapping": {"perf_counter_ns": 0}}),
		encoding="utf-8",
	)
	for serial in [
		BaslerCamera.TOPDOWN.value,
		BaslerCamera.SIDE_0.value,
		BaslerCamera.SIDE_1.value,
		BaslerCamera.SIDE_2.value,
		BaslerCamera.SIDE_3.value,
	]:
		(mocap_raw / f"{serial}.mp4").write_bytes(b"\x00")

	for name in ("eye0.mp4", "eye1.mp4", "world.mp4"):
		(eye_videos / name).write_bytes(b"\x00")

	_write_calibration_toml(calibration / "session_test_camera_calibration.toml")
	return session


def _build_synchronized_session(tmp_path: Path) -> Path:
	"""Session that passes RecordingFolder.check_synchronization."""
	session = _build_minimal_raw_session(tmp_path)
	full_recording = session / "full_recording"
	sync_dir = full_recording / "mocap_data" / "synchronized_corrected_videos"
	sync_dir.mkdir(parents=True)

	for serial in [
		BaslerCamera.TOPDOWN.value,
		BaslerCamera.SIDE_0.value,
		BaslerCamera.SIDE_1.value,
		BaslerCamera.SIDE_2.value,
		BaslerCamera.SIDE_3.value,
	]:
		(sync_dir / f"{serial}.mp4").write_bytes(b"\x00")
		np.save(sync_dir / f"{serial}_timestamps_utc.npy", np.array([0, 1], dtype=np.int64))

	eye_videos = full_recording / "eye_data" / "eye_videos"
	for name in ("eye0", "eye1"):
		np.save(eye_videos / f"{name}_timestamps_utc.npy", np.array([0, 1], dtype=np.int64))

	return session


def test_resolve_session_paths_from_session_root(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	root, full_recording = resolve_session_paths(session)
	assert root == session
	assert full_recording == session / "full_recording"


def test_resolve_session_paths_from_full_recording(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	root, full_recording = resolve_session_paths(session / "full_recording")
	assert root == session
	assert full_recording == session / "full_recording"


def test_validate_raw_session_passes(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	report = validate_session(session, check_sync=False)
	assert report.ok
	assert not report.errors
	assert any(item.name == "pupil_world_video" for item in report.items)


def test_validate_missing_calibration_fails(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	toml_path = session / "calibration" / "session_test_camera_calibration.toml"
	toml_path.unlink()
	report = validate_session(session, check_sync=False)
	assert not report.ok
	assert any(item.name == "calibration_toml" for item in report.errors)


def test_validate_insufficient_cameras_fails(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	toml_path = session / "calibration" / "session_test_camera_calibration.toml"
	_write_calibration_toml(toml_path, n_cams=2)
	report = validate_session(session, check_sync=False)
	assert not report.ok
	assert any(item.name == "calibration_cam_count" for item in report.errors)


def test_validate_synchronized_session_warns_less_on_sync(tmp_path: Path) -> None:
	session = _build_synchronized_session(tmp_path)
	report = validate_session(session, check_sync=True)
	assert report.ok
	sync_items = [item for item in report.items if item.name == "pipeline_sync"]
	assert sync_items
	assert sync_items[0].level == ValidationLevel.INFO
