"""
Tests for python_code/ferret_gaze/

This module tests the gaze pipeline orchestration and its idempotency logic:

- ClipPaths: that paths are derived correctly from a clip directory, and that
  input validation raises when required files are missing.
- eye_kinematics_exists / resampled_data_exists / gaze_kinematics_exists:
  that each returns False when outputs are absent and True when present.
- run_gaze_pipeline skip logic: that pipeline stages are not re-run when their
  outputs already exist (idempotency), and ARE re-run when reprocess flags are set.
- ResamplingStrategy: that the enum values are importable and usable.

The actual numerical computation (eye kinematics, resampling, gaze) is NOT tested
here — those would require real recording data. These tests focus on the
orchestration layer and file-existence checks.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from python_code.ferret_gaze.run_gaze_pipeline import ClipPaths
from python_code.ferret_gaze.data_resampling.data_resampling_helpers import ResamplingStrategy


# =============================================================================
# ClipPaths
# =============================================================================

class TestClipPaths:
    def test_clip_paths_derived_from_root(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.mocap_data_dir == tmp_path / "mocap_data"
        assert paths.eye_data_dir == tmp_path / "eye_data"
        assert paths.analyzable_output_dir == tmp_path / "analyzable_output"
        assert paths.display_videos_dir == tmp_path / "display_videos"

    def test_solver_output_dir(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.solver_output_dir == tmp_path / "mocap_data" / "output_data" / "solver_output"

    def test_gaze_kinematics_output_dir(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.gaze_kinematics_output_dir == tmp_path / "analyzable_output" / "gaze_kinematics"

    def test_blender_script_path(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.blender_script_path.suffix == ".py"

    def test_left_right_eye_name_ferret_757(self, tmp_path):
        clip = tmp_path / "session_757_clip"
        clip.mkdir()
        paths = ClipPaths(clip_path=clip)
        assert paths.left_eye_name == "eye0"
        assert paths.right_eye_name == "eye1"

    def test_left_right_eye_name_other_ferret(self, tmp_path):
        clip = tmp_path / "session_420_clip"
        clip.mkdir()
        paths = ClipPaths(clip_path=clip)
        assert paths.left_eye_name == "eye1"
        assert paths.right_eye_name == "eye0"

    def test_validate_inputs_raises_when_missing(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            paths.validate_inputs()


# =============================================================================
# ClipPaths output-existence checks
# =============================================================================

class TestClipPathsExistenceChecks:
    def test_eye_kinematics_exists_false_when_absent(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.eye_kinematics_exists() is False

    def test_eye_kinematics_exists_true_when_present(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        paths.eye_kinematics_output_dir.mkdir(parents=True)
        for fname in [
            "left_eye_kinematics.csv",
            "left_eye_reference_geometry.json",
            "right_eye_kinematics.csv",
            "right_eye_reference_geometry.json",
        ]:
            (paths.eye_kinematics_output_dir / fname).write_text("{}")
        assert paths.eye_kinematics_exists() is True

    def test_resampled_data_exists_false_when_absent(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.resampled_data_exists() is False

    def test_resampled_data_exists_true_when_present(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        (paths.analyzable_output_dir / "skull_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "left_eye_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "right_eye_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "common_timestamps.npy").write_bytes(b"")
        (paths.analyzable_output_dir / "skull_kinematics" / "skull_kinematics.csv").write_text("")
        (paths.analyzable_output_dir / "left_eye_kinematics" / "left_eye_kinematics.csv").write_text("")
        (paths.analyzable_output_dir / "right_eye_kinematics" / "right_eye_kinematics.csv").write_text("")
        assert paths.resampled_data_exists() is True

    def test_gaze_kinematics_exists_false_when_absent(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.gaze_kinematics_exists() is False

    def test_gaze_kinematics_exists_true_when_present(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        paths.gaze_kinematics_output_dir.mkdir(parents=True)
        for fname in [
            "left_gaze_kinematics.csv",
            "left_gaze_reference_geometry.json",
            "right_gaze_kinematics.csv",
            "right_gaze_reference_geometry.json",
        ]:
            (paths.gaze_kinematics_output_dir / fname).write_text("{}")
        assert paths.gaze_kinematics_exists() is True

    def test_blender_script_exists_false_when_absent(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        assert paths.blender_script_exists() is False

    def test_blender_script_exists_true_when_present(self, tmp_path):
        paths = ClipPaths(clip_path=tmp_path)
        paths.blender_script_path.parent.mkdir(parents=True, exist_ok=True)
        paths.blender_script_path.write_text("# blender script")
        assert paths.blender_script_exists() is True


# =============================================================================
# run_gaze_pipeline skip logic
# =============================================================================

class TestGazePipelineSkipLogic:
    """
    Verify that pipeline stages are skipped when outputs exist and that
    reprocess flags correctly bypass the skip.
    Uses mock patches so no actual computation runs.
    """

    def _patch_all_stages(self):
        return [
            patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_eye_kinematics"),
            patch("python_code.ferret_gaze.run_gaze_pipeline.build_video_configs", return_value=[]),
            patch("python_code.ferret_gaze.run_gaze_pipeline.resample_all_data"),
            patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_gaze"),
            patch("python_code.ferret_gaze.run_gaze_pipeline.generate_blender_script"),
            patch("python_code.ferret_gaze.run_gaze_pipeline.ClipPaths.validate_inputs"),
        ]

    def _populate_all_outputs(self, paths: ClipPaths):
        """Write stub output files so all existence checks return True."""
        paths.eye_kinematics_output_dir.mkdir(parents=True)
        for fname in [
            "left_eye_kinematics.csv", "left_eye_reference_geometry.json",
            "right_eye_kinematics.csv", "right_eye_reference_geometry.json",
        ]:
            (paths.eye_kinematics_output_dir / fname).write_text("{}")

        (paths.analyzable_output_dir / "skull_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "left_eye_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "right_eye_kinematics").mkdir(parents=True)
        (paths.analyzable_output_dir / "common_timestamps.npy").write_bytes(b"")
        (paths.analyzable_output_dir / "skull_kinematics" / "skull_kinematics.csv").write_text("")
        (paths.analyzable_output_dir / "left_eye_kinematics" / "left_eye_kinematics.csv").write_text("")
        (paths.analyzable_output_dir / "right_eye_kinematics" / "right_eye_kinematics.csv").write_text("")

        paths.gaze_kinematics_output_dir.mkdir(parents=True)
        for fname in [
            "left_gaze_kinematics.csv", "left_gaze_reference_geometry.json",
            "right_gaze_kinematics.csv", "right_gaze_reference_geometry.json",
        ]:
            (paths.gaze_kinematics_output_dir / fname).write_text("{}")

        paths.blender_script_path.parent.mkdir(parents=True, exist_ok=True)
        paths.blender_script_path.write_text("# blender")

    def test_all_stages_skipped_when_outputs_exist(self, tmp_path):
        from python_code.ferret_gaze.run_gaze_pipeline import run_gaze_pipeline
        paths = ClipPaths(clip_path=tmp_path)
        self._populate_all_outputs(paths)

        with patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_eye_kinematics") as mock_eye, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.resample_all_data") as mock_resample, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_gaze") as mock_gaze, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.generate_blender_script") as mock_blender, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.ClipPaths.validate_inputs"):
            run_gaze_pipeline(recording_path=tmp_path)

        mock_eye.assert_not_called()
        mock_resample.assert_not_called()
        mock_gaze.assert_not_called()
        mock_blender.assert_not_called()

    def test_reprocess_all_calls_all_stages(self, tmp_path):
        from python_code.ferret_gaze.run_gaze_pipeline import run_gaze_pipeline
        paths = ClipPaths(clip_path=tmp_path)
        self._populate_all_outputs(paths)

        with patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_eye_kinematics") as mock_eye, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.build_video_configs", return_value=[]) as mock_vids, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.resample_all_data") as mock_resample, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.calculate_gaze") as mock_gaze, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.generate_blender_script") as mock_blender, \
             patch("python_code.ferret_gaze.run_gaze_pipeline.ClipPaths.validate_inputs"):
            run_gaze_pipeline(recording_path=tmp_path, reprocess_all=True)

        mock_eye.assert_called_once()
        mock_resample.assert_called_once()
        mock_gaze.assert_called_once()
        mock_blender.assert_called_once()


# =============================================================================
# ResamplingStrategy
# =============================================================================

class TestResamplingStrategy:
    def test_fastest_strategy_importable(self):
        assert ResamplingStrategy.FASTEST is not None

    def test_all_enum_values_exist(self):
        values = [s for s in ResamplingStrategy]
        assert len(values) > 0
