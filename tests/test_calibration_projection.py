"""Tests for session calibration TOML -> OpenCV projection matrices (P2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import toml

from src.ferret_gaze.realtime.calibration_projection import (
    SessionMultiViewCalibration,
    discover_session_calibration_toml,
    load_session_multi_view_calibration,
    projection_matrix_from_cam_block,
)
from src.ferret_gaze.realtime.runtime_config import RealtimeRuntimeConfig


def test_projection_identity_camera_at_origin() -> None:
    block = {
        "name": "cam_a",
        "world_position": [0.0, 0.0, 2.0],
        "world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
    }
    p = projection_matrix_from_cam_block(block)
    assert p.shape == (3, 4)
    k = np.array(block["matrix"], dtype=np.float64)
    t = np.array([[0.0], [0.0], [-2.0]])
    expected = k @ np.hstack([np.eye(3), t])
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-9)


def test_projection_rejects_bad_matrix_shape() -> None:
    block = {
        "world_position": [0, 0, 0],
        "world_orientation": np.eye(3).tolist(),
        "matrix": [[1, 0], [0, 1]],
    }
    with pytest.raises(ValueError, match="matrix"):
        projection_matrix_from_cam_block(block)


def test_load_multi_view_minimal_toml(tmp_path: Path) -> None:
    data = {
        "cam_1": {
            "name": "side",
            "world_position": [1.0, 0.0, 1.0],
            "world_orientation": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "matrix": [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
        },
        "cam_0": {
            "name": "top",
            "world_position": [0.0, 0.0, 2.0],
            "world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        },
    }
    path = tmp_path / "session_calibration_camera_calibration.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)

    calib = load_session_multi_view_calibration(path)
    assert isinstance(calib, SessionMultiViewCalibration)
    assert set(calib.projection_by_cam_index.keys()) == {0, 1}
    assert calib.camera_name_by_index[0] == "top"
    assert calib.camera_name_by_index[1] == "side"
    assert calib.projection_matrix(0).shape == (3, 4)


def test_discover_session_calibration_toml(tmp_path: Path) -> None:
    session = tmp_path / "session_fake"
    cal = session / "calibration"
    cal.mkdir(parents=True)
    toml_path = cal / "foo_camera_calibration.toml"
    toml_path.write_text(
        '[cam_0]\nname="x"\nworld_position=[0,0,0]\n'
        'world_orientation=[[1,0,0],[0,1,0],[0,0,1]]\n'
        'matrix=[[1,0,0],[0,1,0],[0,0,1]]\n',
        encoding="utf-8",
    )
    found = discover_session_calibration_toml(session)
    assert found == toml_path


def test_discover_returns_none_without_calibration_dir(tmp_path: Path) -> None:
    assert discover_session_calibration_toml(tmp_path / "no_cal") is None


def test_realtime_runtime_config_calibration_path_optional() -> None:
    cfg = RealtimeRuntimeConfig(calibration_toml_path="/tmp/example_camera_calibration.toml")
    assert cfg.calibration_toml_path == "/tmp/example_camera_calibration.toml"


def test_load_rejects_toml_without_cam_tables(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text('other = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cam_\\*"):
        load_session_multi_view_calibration(path)
