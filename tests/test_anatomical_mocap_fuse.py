"""Tests for anatomical gaze fusion and binocular rolling calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.anatomical_mocap_fuse import (
    AnatomicalMocapGazeFuser,
    AnatomicalRollingEyeCalibrator,
    create_eye_calibrator,
    create_gaze_fuser,
)
from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    RollingCalibrationState,
    TriangulationResult,
)


def test_create_factories_stub_defaults() -> None:
    assert create_eye_calibrator("stub").__class__.__name__ == "StubRollingEyeCalibrator"
    assert create_gaze_fuser("stub").__class__.__name__ == "StubGazeFuser"


def test_create_factories_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported eye_calibrator"):
        create_eye_calibrator("ceres")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported gaze_fuser"):
        create_gaze_fuser("ceres")  # type: ignore[arg-type]


def test_anatomical_fuser_places_eyes_in_world_for_identity_skull() -> None:
    """Skull at origin, identity quaternion: eye origins match skull-local offsets."""
    packet = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(0.0, 0.0, 1.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 0.0, 1.0),
        confidence=1.0,
    )
    tri = TriangulationResult(seq=0, skull_position_xyz=(0.0, 0.0, 0.0))
    cal = RollingCalibrationState(gain=1.0, offset=0.0)
    inf = FrameInferenceResult(seq=0, confidence=0.9)
    fuser = AnatomicalMocapGazeFuser(half_ipd_mm=10.0, eye_y_mm=25.0, eye_z_mm=0.0)
    out = fuser.fuse(packet, tri, cal, inf)
    assert out.left_eye_origin_xyz == pytest.approx((-10.0, 25.0, 0.0))
    assert out.right_eye_origin_xyz == pytest.approx((10.0, 25.0, 0.0))
    for d in (out.left_gaze_direction_xyz, out.right_gaze_direction_xyz):
        assert math.isclose(float(np.linalg.norm(d)), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_anatomical_fuser_rotates_eye_origins_with_skull_orientation() -> None:
    """90 deg yaw around +Y maps skull -X rest toward world +Z (right-handed)."""
    r = Rotation.from_euler("y", 90.0, degrees=True)
    qx, qy, qz, qw = r.as_quat()
    wxyz = (float(qw), float(qx), float(qy), float(qz))
    packet = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=wxyz,
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(0.0, 0.0, 1.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 0.0, 1.0),
        confidence=1.0,
    )
    tri = TriangulationResult(seq=0, skull_position_xyz=(100.0, 0.0, 0.0))
    cal = RollingCalibrationState(gain=1.0, offset=0.0)
    inf = FrameInferenceResult(seq=0, confidence=1.0)
    fuser = AnatomicalMocapGazeFuser(half_ipd_mm=10.0, eye_y_mm=0.0, eye_z_mm=0.0)
    out = fuser.fuse(packet, tri, cal, inf)
    # Left rest (-10,0,0) in skull -> approximately (0,0,+10) offset in world after +90Y.
    assert out.left_eye_origin_xyz[0] == pytest.approx(100.0, abs=0.05)
    assert out.left_eye_origin_xyz[2] == pytest.approx(10.0, abs=0.05)


def _yaw_xz(d: tuple[float, float, float]) -> float:
    x, y, z = d
    return float(math.atan2(x, z))


def test_anatomical_calibrator_reduces_skull_frame_yaw_disparity() -> None:
    """Closed loop: EMA biases + fuse drive horizontal yaw difference toward zero."""
    tri = TriangulationResult(seq=0, skull_position_xyz=(0.0, 0.0, 0.0))
    inf = FrameInferenceResult(seq=0, confidence=1.0)
    calibrator = AnatomicalRollingEyeCalibrator(vergence_ema_alpha=0.25, max_bias_rad=2.0)
    fuser = AnatomicalMocapGazeFuser(half_ipd_mm=10.0, eye_y_mm=25.0, eye_z_mm=0.0)
    packet = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(1.0, 0.0, 0.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 0.0, 1.0),
        confidence=1.0,
    )
    r_ws = np.eye(3)
    ls0 = r_ws.T @ np.array(packet.left_gaze_direction_xyz, dtype=np.float64)
    rs0 = r_ws.T @ np.array(packet.right_gaze_direction_xyz, dtype=np.float64)
    start_diff = abs(_yaw_xz(tuple(ls0)) - _yaw_xz(tuple(rs0)))
    p = packet
    for _ in range(80):
        st = calibrator.update(tri, inference=inf, packet=p)
        p = fuser.fuse(p, tri, st, inf)
        lw = np.array(p.left_gaze_direction_xyz, dtype=np.float64)
        rw = np.array(p.right_gaze_direction_xyz, dtype=np.float64)
        ls = r_ws.T @ lw
        rs = r_ws.T @ rw
    end_diff = abs(_yaw_xz(tuple(ls)) - _yaw_xz(tuple(rs)))
    assert end_diff < start_diff * 0.5
