"""Tests for linear DLT multi-view triangulation (P3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import toml

from src.ferret_gaze.realtime.calibration_projection import (
    SessionMultiViewCalibration,
    load_session_multi_view_calibration,
)
from src.ferret_gaze.realtime.multiview_triangulation import triangulate_linear_dlt
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    MultiviewOpenCvTriangulator,
    create_triangulator,
)


def _pixel_from_p(p: np.ndarray, x_world: np.ndarray) -> np.ndarray:
    """Project inhomogeneous world point with 3x4 ``P`` (OpenCV convention)."""
    x_h = np.array([x_world[0], x_world[1], x_world[2], 1.0], dtype=np.float64)
    y = p @ x_h
    return y[:2] / y[2]


def test_triangulate_linear_dlt_two_views_recover_ground_truth() -> None:
    x_gt = np.array([0.15, -0.08, 1.25], dtype=np.float64)
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p0 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    c1 = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    r1 = np.eye(3, dtype=np.float64)
    t1 = (-r1 @ c1).reshape(3, 1)
    p1 = k @ np.hstack([r1, t1])
    uv0 = _pixel_from_p(p0, x_gt)
    uv1 = _pixel_from_p(p1, x_gt)
    x_hat = triangulate_linear_dlt([p0, p1], [uv0, uv1])
    np.testing.assert_allclose(x_hat, x_gt, rtol=1e-5, atol=1e-5)


def test_triangulate_linear_dlt_rejects_single_view() -> None:
    p = np.eye(3, 4, dtype=np.float64)
    with pytest.raises(ValueError, match="two"):
        triangulate_linear_dlt([p], [np.array([1.0, 2.0])])


def test_multiview_triangulator_round_trip_with_loaded_toml(tmp_path: Path) -> None:
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
    tri = MultiviewOpenCvTriangulator(calibration=calib)
    x_gt = np.array([0.05, 0.05, 0.9], dtype=np.float64)
    p0 = calib.projection_matrix(0)
    p1 = calib.projection_matrix(1)
    uv0 = _pixel_from_p(p0, x_gt)
    uv1 = _pixel_from_p(p1, x_gt)
    inference = FrameInferenceResult(
        seq=7,
        confidence=1.0,
        single_landmark_uv_by_cam=((0, float(uv0[0]), float(uv0[1])), (1, float(uv1[0]), float(uv1[1]))),
    )
    out = tri.triangulate(inference)
    assert out.seq == 7
    np.testing.assert_allclose(np.array(out.skull_position_xyz, dtype=np.float64), x_gt, rtol=1e-4, atol=1e-4)


def test_multiview_triangulator_falls_back_with_one_view() -> None:
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p0 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])

    calib = SessionMultiViewCalibration(
        projection_by_cam_index={0: p0},
        camera_name_by_index={0: "only"},
    )
    tri = MultiviewOpenCvTriangulator(calibration=calib)
    inference = FrameInferenceResult(
        seq=1,
        confidence=1.0,
        keypoints_xyz=((1.0, 2.0, 3.0), (5.0, 5.0, 5.0)),
        single_landmark_uv_by_cam=((0, 100.0, 200.0),),
    )
    out = tri.triangulate(inference)
    # Centroid of keypoints_xyz: ((1+5)/2, (2+5)/2, (3+5)/2) = (3, 3.5, 4)
    assert out.skull_position_xyz == (3.0, 3.5, 4.0)


def test_create_triangulator_multiview_requires_calibration_path() -> None:
    with pytest.raises(ValueError, match="calibration_toml_path"):
        create_triangulator(backend="multiview_opencv")


def test_create_triangulator_multiview_opencv(tmp_path: Path) -> None:
    data = {
        "cam_0": {
            "name": "a",
            "world_position": [0.0, 0.0, 2.0],
            "world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        },
        "cam_1": {
            "name": "b",
            "world_position": [0.3, 0.0, 2.0],
            "world_orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        },
    }
    path = tmp_path / "session_calibration_camera_calibration.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)

    tri = create_triangulator(backend="multiview_opencv", calibration_toml_path=path)
    assert isinstance(tri, MultiviewOpenCvTriangulator)
