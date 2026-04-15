"""Tests for Kabsch skull solver and SE(3) helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.kabsch_skull_solver import (
    DEFAULT_KABSCH_REFERENCE_BODY,
    KabschRealtimeSkullSolver,
    create_skull_solver,
    kabsch_rotation_translation,
    load_kabsch_reference_body,
)
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    RealtimeInferenceRuntime,
    StubGazeFuser,
    StubRollingEyeCalibrator,
    StubTriangulator,
    TriangulationResult,
)
from src.ferret_gaze.realtime.live_mocap_pipeline import process_live_mocap_tick
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet


def test_kabsch_rotation_identity() -> None:
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    r, t = kabsch_rotation_translation(ref, ref)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(t, 0.0, atol=1e-9)


def test_kabsch_rotation_recovers_known_orientation() -> None:
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    r_true = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
    obs = ref @ r_true.T
    r_hat, t_hat = kabsch_rotation_translation(obs, ref)
    np.testing.assert_allclose(r_hat, r_true, atol=1e-5)
    np.testing.assert_allclose(t_hat, 0.0, atol=1e-5)


def test_create_skull_solver_none_returns_none() -> None:
    assert create_skull_solver("none") is None
    assert create_skull_solver("") is None


def test_create_skull_solver_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_skull_solver("not_a_solver_backend")  # type: ignore[arg-type]


def test_load_kabsch_reference_body_round_trip(tmp_path: Path) -> None:
    ref = np.arange(12, dtype=np.float64).reshape(4, 3)
    path = tmp_path / "ref.npy"
    np.save(path, ref)
    loaded = load_kabsch_reference_body(path)
    np.testing.assert_allclose(loaded, ref)


def test_kabsch_solver_updates_quaternion_in_process_tick() -> None:
    """End-to-end: three observed keypoints = rotated default template -> fused quat matches."""
    r_true = Rotation.from_euler("z", 40.0, degrees=True).as_matrix()
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    obs = ref @ r_true.T
    keypoints_xyz = tuple((float(x), float(y), float(z)) for x, y, z in obs)

    class _FixedInfer(RealtimeInferenceRuntime):
        def infer(self, packet, *, frame_set=None):
            return FrameInferenceResult(
                seq=packet.seq,
                confidence=1.0,
                keypoints_xyz=keypoints_xyz,
                single_landmark_uv_by_cam=(),
            )

    fs = LiveMocapFrameSet(seq=0, anchor_utc_ns=1_000, images_bgr={0: np.zeros((2, 2, 3), dtype=np.uint8)})
    solver = KabschRealtimeSkullSolver(reference_body=ref)
    fused = process_live_mocap_tick(
        fs,
        inference_runtime=_FixedInfer(),
        triangulator=StubTriangulator(),
        calibrator=StubRollingEyeCalibrator(),
        fuser=StubGazeFuser(),
        skull_solver=solver,
    )
    q_out = np.array(fused.skull_quaternion_wxyz, dtype=np.float64)
    q_exp = Rotation.from_matrix(r_true).as_quat()
    r_out = Rotation.from_quat([q_out[1], q_out[2], q_out[3], q_out[0]])
    r_exp = Rotation.from_quat(q_exp)
    ang = (r_out.inv() * r_exp).magnitude()
    assert ang < 0.02


def test_kabsch_solver_skips_when_fewer_than_three_keypoints() -> None:
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    solver = KabschRealtimeSkullSolver(reference_body=ref)

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
    tri = TriangulationResult(seq=0, skull_position_xyz=(0.5, 0.5, 0.5))
    inf = FrameInferenceResult(
        seq=0,
        confidence=1.0,
        keypoints_xyz=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        single_landmark_uv_by_cam=(),
    )
    out = solver.solve_with_context(packet, inference=inf, triangulated=tri)
    assert out.skull_quaternion_wxyz == packet.skull_quaternion_wxyz
