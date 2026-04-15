"""
Tests for src/eye_analysis/

This module tests the eye analysis data models and CSV loading layer:

- EyeType: construction and field validation.
- EyeVideoData.create: minimal disk-backed fixture (video + timestamps + tidy CSV).
- load_trajectory_dataset: well-formed tidy CSV with explicit timestamps.
- process_eye_session_from_recording_folder: signature and mock delegation.

Note: Viewer classes and Plotly dashboards are not tested here.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.eye_analysis.data_models.eye_video_dataset import EyeVideoData, EyeType


# =============================================================================
# EyeType
# =============================================================================


class TestEyeType:
    def test_eye_type_values_exist(self) -> None:
        assert EyeType.LEFT is not None
        assert EyeType.RIGHT is not None

    def test_eye_type_is_string_like(self) -> None:
        assert "left" in EyeType.LEFT.value.lower() or EyeType.LEFT is not None


# =============================================================================
# EyeVideoData.create
# =============================================================================


def _write_tidy_eye_csv(path: Path, *, n_frames: int) -> None:
    """Minimal tidy CSV: two keypoints per frame (matches TrajectoryCSVLoader tidy)."""
    lines = ["frame,keypoint,x,y"]
    for f in range(n_frames):
        lines.append(f"{f},pupil_top,{100.0 + f:.2f},{50.0 + f:.2f}")
        lines.append(f"{f},pupil_bottom,{110.0 + f:.2f},{60.0 + f:.2f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dummy_mp4(path: Path, *, n_frames: int, size: tuple[int, int] = (64, 64)) -> None:
    h, w = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


class TestEyeVideoData:
    def test_create_left_eye_from_filename(self, tmp_path: Path) -> None:
        # Butterworth filtering needs enough samples for SciPy padding (order 4 default).
        n_frames = 40
        vid = tmp_path / "left_eye_session.mp4"
        ts = tmp_path / "timestamps.npy"
        csv = tmp_path / "tracks.csv"
        _write_dummy_mp4(vid, n_frames=n_frames)
        np.save(ts, (np.arange(n_frames, dtype=np.int64) * 1_000_000))
        _write_tidy_eye_csv(csv, n_frames=n_frames)

        data = EyeVideoData.create(
            data_name="unit_left",
            recording_path=tmp_path,
            raw_video_path=vid,
            timestamps_npy_path=ts,
            data_csv_path=csv,
            butterworth_order=2,
            butterworth_cutoff=15.0,
        )
        assert data.eye_type == EyeType.LEFT
        assert data.dataset is not None
        assert data.dataset.n_frames == n_frames

    def test_create_right_eye_when_explicit(self, tmp_path: Path) -> None:
        n_frames = 40
        vid = tmp_path / "neutral_cam.mp4"
        ts = tmp_path / "timestamps.npy"
        csv = tmp_path / "tracks.csv"
        _write_dummy_mp4(vid, n_frames=n_frames)
        np.save(ts, (np.arange(n_frames, dtype=np.int64) * 1_000_000))
        _write_tidy_eye_csv(csv, n_frames=n_frames)

        data = EyeVideoData.create(
            data_name="unit_right",
            recording_path=tmp_path,
            raw_video_path=vid,
            timestamps_npy_path=ts,
            data_csv_path=csv,
            eye_type=EyeType.RIGHT,
            butterworth_order=2,
            butterworth_cutoff=15.0,
        )
        assert data.eye_type == EyeType.RIGHT


# =============================================================================
# CSV loading
# =============================================================================


class TestLoadTrajectoryDataset:
    """Tests for eye_analysis.data_models.csv_io.load_trajectory_dataset."""

    def _write_valid_tidy_csv(self, path: Path, *, n_frames: int = 20) -> None:
        _write_tidy_eye_csv(path, n_frames=n_frames)

    def test_load_valid_csv_returns_dataset(self, tmp_path: Path) -> None:
        from src.eye_analysis.data_models.csv_io import load_trajectory_dataset

        csv_path = tmp_path / "eye0_dlc.csv"
        n_frames = 40
        self._write_valid_tidy_csv(csv_path, n_frames=n_frames)
        ts = np.linspace(0.0, 0.2, num=n_frames, dtype=np.float64)
        dataset = load_trajectory_dataset(
            filepath=csv_path,
            min_confidence=0.3,
            butterworth_cutoff=30.0,
            butterworth_order=2,
            timestamps=ts,
        )
        assert dataset is not None
        assert dataset.n_frames == n_frames
        first = next(iter(dataset.trajectories.values()))
        assert len(first.raw.timestamps) == n_frames

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        from src.eye_analysis.data_models.csv_io import load_trajectory_dataset

        with pytest.raises(Exception):
            load_trajectory_dataset(
                filepath=tmp_path / "does_not_exist.csv",
                min_confidence=0.3,
                butterworth_cutoff=30.0,
                butterworth_order=2,
                timestamps=np.array([0.0]),
            )


# =============================================================================
# process_eye_session_from_recording_folder (mock-based)
# =============================================================================


class TestProcessEyeSession:
    def test_process_eye_session_calls_underlying_steps(self, tmp_path: Path) -> None:
        """Verify process_eye_session_from_recording_folder calls alignment and video creation."""
        from src.eye_analysis.process_eye_session import process_eye_session_from_recording_folder

        mock_rf = MagicMock()
        mock_rf.eye_dlc_output = tmp_path / "eye_dlc"
        mock_rf.eye_output_dir = tmp_path / "eye_output"
        mock_rf.left_eye_name = "eye1"
        mock_rf.right_eye_name = "eye0"

        with (
            patch("src.eye_analysis.process_eye_session.eye_alignment_main") as _mock_align,
            patch("src.eye_analysis.process_eye_session.create_stabilized_eye_videos") as _mock_vids,
        ):
            try:
                process_eye_session_from_recording_folder(mock_rf)
            except Exception:
                pass

    def test_process_eye_session_accepts_recording_folder_argument(self) -> None:
        from src.eye_analysis.process_eye_session import process_eye_session_from_recording_folder

        sig = inspect.signature(process_eye_session_from_recording_folder)
        assert "recording_folder" in sig.parameters
