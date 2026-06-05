"""Tests for session path discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.utilities.folder_utilities.session_paths import (
	discover_basler_raw_videos_folder,
	discover_calibration_toml,
	discover_pupil_info_json,
	discover_pupil_output_folder,
)


def _build_base_data_session(tmp_path: Path) -> Path:
	"""Layout matching session_2026-03-19_psychopy_trial_1_ferret411."""
	session = tmp_path / "session_2026-03-19_psychopy_trial_1_ferret411"
	base_data = session / "base_data"
	pupil_output = base_data / "pupil_output"
	raw_videos = base_data / "raw_videos"
	pupil_output.mkdir(parents=True)
	raw_videos.mkdir(parents=True)
	(session / "full_recording" / "mocap_data").mkdir(parents=True)
	(session / "full_recording" / "eye_data").mkdir(parents=True)

	(pupil_output / "info.player.json").write_text(
		json.dumps({"start_time_synced_s": 0.0, "start_time_system_s": 0.0}),
		encoding="utf-8",
	)
	for name in ("eye0.mp4", "eye1.mp4", "world.mp4"):
		(pupil_output / name).write_bytes(b"\x00")
	(raw_videos / "timestamp_mapping.json").write_text("{}", encoding="utf-8")
	return session


def test_discover_pupil_output_under_base_data(tmp_path: Path) -> None:
	session = _build_base_data_session(tmp_path)
	assert discover_pupil_output_folder(session) == session / "base_data" / "pupil_output"


def test_discover_pupil_info_json_under_base_data(tmp_path: Path) -> None:
	session = _build_base_data_session(tmp_path)
	info_path = discover_pupil_info_json(session)
	assert info_path == session / "base_data" / "pupil_output" / "info.player.json"


def test_discover_basler_raw_videos_under_base_data(tmp_path: Path) -> None:
	session = _build_base_data_session(tmp_path)
	assert discover_basler_raw_videos_folder(session) == session / "base_data" / "raw_videos"


def test_discover_calibration_toml_from_explicit_path(tmp_path: Path) -> None:
	session = _build_base_data_session(tmp_path)
	other_session = tmp_path / "prior_calibration_session"
	cal_dir = other_session / "calibration"
	cal_dir.mkdir(parents=True)
	toml_path = cal_dir / "session_prior_camera_calibration.toml"
	toml_path.write_text('[cam_0]\nname="a"\n', encoding="utf-8")
	assert discover_calibration_toml(session, explicit_path=toml_path) == toml_path


def test_discover_calibration_toml_missing_returns_none(tmp_path: Path) -> None:
	session = _build_base_data_session(tmp_path)
	assert discover_calibration_toml(session) is None
