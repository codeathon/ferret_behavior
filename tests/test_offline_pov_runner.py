"""Tests for offline POV CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.offline_pov.pipeline_config import OfflinePipelineConfig
from src.offline_pov.run_offline_pov import copy_clip_for_local_inspection, run_from_config
from tests.test_validate_session import _build_minimal_raw_session


def test_run_from_config_validate_only(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	config = OfflinePipelineConfig(session_root=session)
	code = run_from_config(config, validate_only=True)
	assert code == 0


def test_run_from_config_pipeline_fails_without_calibration_toml(tmp_path: Path) -> None:
	from tests.test_session_paths import _build_base_data_session

	session = _build_base_data_session(tmp_path)
	(session / "calibration").mkdir(parents=True)
	config = OfflinePipelineConfig(session_root=session)
	code = run_from_config(config, skip_gaze=True, skip_copy=True)
	assert code == 1


def test_run_from_config_runs_pipeline_and_gaze(tmp_path: Path) -> None:
	session = _build_minimal_raw_session(tmp_path)
	clip = session / "clips" / "test_clip"
	clip.mkdir(parents=True)
	(clip / "mocap_data").mkdir()
	(clip / "eye_data").mkdir()
	(clip / "analyzable_output").mkdir()
	(clip / "display_videos").mkdir()
	(clip / "analyzable_output" / "ferret_full_gaze_blender_viz.py").write_text("# stub")

	config = OfflinePipelineConfig(
		session_root=session,
		clip_name="test_clip",
		local_inspection_dir=tmp_path / "inspect",
	)

	with patch("src.offline_pov.run_offline_pov.run_full_offline_pipeline") as mock_pipeline, \
		patch("src.offline_pov.run_offline_pov.run_gaze_pipeline", return_value=clip / "analyzable_output") as mock_gaze:
		code = run_from_config(config, skip_pipeline=False, skip_gaze=False, skip_copy=False)

	assert code == 0
	mock_pipeline.assert_called_once()
	mock_gaze.assert_called_once()
	assert (tmp_path / "inspect" / "test_clip" / "analyzable_output").is_dir()


def test_copy_clip_for_local_inspection(tmp_path: Path) -> None:
	clip = tmp_path / "clip_a"
	(clip / "analyzable_output").mkdir(parents=True)
	(clip / "display_videos").mkdir(parents=True)
	(clip / "analyzable_output" / "gaze.csv").write_text("frame\n0\n")
	dest = tmp_path / "local_inspect"
	target = copy_clip_for_local_inspection(clip, dest)
	assert target == dest / "clip_a"
	assert (target / "analyzable_output" / "gaze.csv").read_text() == "frame\n0\n"
	assert (target / "README.txt").exists()


def test_load_config_from_json(tmp_path: Path) -> None:
	config_path = tmp_path / "offline_pipeline.json"
	config_path.write_text(
		json.dumps(
			{
				"session_root": "/tmp/session",
				"external_tools": {
					"skellyclicker_python": "/custom/python",
					"skellyclicker_script": "/custom/script.py",
					"triangulation_python": "/tri/python",
					"triangulation_script": "/tri/script.py",
					"calibration_python": "/cal/python",
					"calibration_script": "/cal/script.py",
				},
			}
		),
		encoding="utf-8",
	)
	config = OfflinePipelineConfig.from_json_file(config_path)
	assert config.external_tools.skellyclicker_python == "/custom/python"
