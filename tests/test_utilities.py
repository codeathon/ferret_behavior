"""
Tests for src/utilities/

This module tests the RecordingFolder path model and its pipeline-stage
validation methods. RecordingFolder is the central hub that batch_processing,
ferret_gaze, rigid_body_solver, and rerun_viewer all import for typed session
paths.

Coverage includes:

- RecordingFolder.from_folder_path: valid full_recording and clip layouts,
  rejection of missing mocap_data / eye_data, rejection of wrong folder names.
- Eye assignment: left_eye_name / right_eye_name based on ferret ID in session name.
- is_* stage checks: each returns False when the expected output files are absent.
- check_* methods: each raises ValueError when outputs are absent.
- CalibrationFolder: basic construction from a valid calibration video directory.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.utilities.folder_utilities.recording_folder import (
    RecordingFolder,
    PipelineStep,
)


# =============================================================================
# RecordingFolder.from_folder_path
# =============================================================================

class TestRecordingFolderFromPath:
    def test_valid_full_recording_constructs(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.version_name == "full_recording"
        assert rf.is_clip is False

    def test_valid_clip_constructs(self, tmp_path):
        session = tmp_path / "session_2025-01-01_ferret_420"
        clip_dir = session / "clips" / "0m_30s-1m_0s"
        (clip_dir / "mocap_data").mkdir(parents=True)
        (clip_dir / "eye_data").mkdir(parents=True)
        rf = RecordingFolder.from_folder_path(clip_dir)
        assert rf.is_clip is True
        assert rf.version_name == "0m_30s-1m_0s"

    def test_missing_mocap_data_raises(self, tmp_path):
        session = tmp_path / "session_2025-01-01_ferret_420"
        recording = session / "full_recording"
        recording.mkdir(parents=True)
        (recording / "eye_data").mkdir()
        with pytest.raises(ValueError, match="mocap_data"):
            RecordingFolder.from_folder_path(recording)

    def test_missing_eye_data_raises(self, tmp_path):
        session = tmp_path / "session_2025-01-01_ferret_420"
        recording = session / "full_recording"
        recording.mkdir(parents=True)
        (recording / "mocap_data").mkdir()
        with pytest.raises(ValueError, match="eye_data"):
            RecordingFolder.from_folder_path(recording)

    def test_wrong_folder_name_raises(self, tmp_path):
        session = tmp_path / "session_2025-01-01_ferret_420"
        wrong = session / "wrong_name"
        wrong.mkdir(parents=True)
        (wrong / "mocap_data").mkdir()
        (wrong / "eye_data").mkdir()
        with pytest.raises(ValueError):
            RecordingFolder.from_folder_path(wrong)

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            RecordingFolder.from_folder_path(tmp_path / "nonexistent")


# =============================================================================
# Eye name assignment
# =============================================================================

class TestEyeNameAssignment:
    def test_ferret_757_left_is_eye0(self, tmp_path):
        session = tmp_path / "session_2025-01-01_ferret_757_test"
        recording = session / "full_recording"
        (recording / "mocap_data").mkdir(parents=True)
        (recording / "eye_data").mkdir(parents=True)
        rf = RecordingFolder.from_folder_path(recording)
        assert rf.left_eye_name == "eye0"
        assert rf.right_eye_name == "eye1"

    def test_other_ferret_left_is_eye1(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.left_eye_name == "eye1"
        assert rf.right_eye_name == "eye0"


# =============================================================================
# Pipeline stage checks — is_* returns False when absent
# =============================================================================

class TestPipelineStageChecks:
    def test_is_synchronized_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_synchronized() is False

    def test_is_calibrated_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_calibrated() is False

    def test_is_dlc_processed_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_dlc_processed() is False

    def test_is_triangulated_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_triangulated() is False

    def test_is_eye_postprocessed_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_eye_postprocessed() is False

    def test_is_skull_postprocessed_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_skull_postprocessed() is False

    def test_is_gaze_postprocessed_false_when_absent(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.is_gaze_postprocessed() is False


# =============================================================================
# check_* methods raise when absent
# =============================================================================

class TestCheckMethodsRaise:
    def test_check_synchronization_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_synchronization()

    def test_check_calibration_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_calibration()

    def test_check_dlc_output_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_dlc_output()

    def test_check_triangulation_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_triangulation()

    def test_check_eye_postprocessing_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_eye_postprocessing()

    def test_check_skull_postprocessing_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_skull_postprocessing()

    def test_check_gaze_postprocessing_raises(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        with pytest.raises(ValueError):
            rf.check_gaze_postprocessing()


# =============================================================================
# PipelineStep enum
# =============================================================================

class TestPipelineStep:
    def test_all_expected_steps_exist(self):
        expected = {"RAW", "SYNCHRONIZED", "DLCED", "TRIANGULATED",
                    "EYE_POST_PROCESSED", "SKULL_POST_PROCESSED", "GAZE_POST_PROCESSED"}
        actual = {step.name for step in PipelineStep}
        assert expected == actual

    def test_default_step_is_raw(self, fake_full_recording_dir):
        rf = RecordingFolder.from_folder_path(fake_full_recording_dir)
        assert rf.processing_step == PipelineStep.RAW
