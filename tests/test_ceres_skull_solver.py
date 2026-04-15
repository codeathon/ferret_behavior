"""Tests for Ceres-style (SciPy) sliding-window SE(3) skull solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.ceres_skull_solver import CeresRealtimeSkullSolver
from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.kabsch_skull_solver import (
    DEFAULT_KABSCH_REFERENCE_BODY,
    KabschRealtimeSkullSolver,
    create_skull_solver,
)
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.live_mocap_pipeline import process_live_mocap_tick
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    RealtimeInferenceRuntime,
    StubGazeFuser,
    StubRollingEyeCalibrator,
    StubTriangulator,
    TriangulationResult,
)


def test_create_skull_solver_ceres_returns_solver() -> None:
    s = create_skull_solver("ceres", ceres_window_size=2)
    assert isinstance(s, CeresRealtimeSkullSolver)
    assert s.name == "ceres_sliding_scipy"


def test_ceres_window_one_matches_kabsch_quaternion() -> None:
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    r_true = Rotation.from_euler("y", 22.0, degrees=True).as_matrix()
    obs = ref @ r_true.T
    kabsch = KabschRealtimeSkullSolver(reference_body=ref)
    ceres = CeresRealtimeSkullSolver(ref, window_size=1, max_nfev=200)
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
    inf = FrameInferenceResult(
        seq=0,
        confidence=1.0,
        keypoints_xyz=tuple((float(x), float(y), float(z)) for x, y, z in obs),
        single_landmark_uv_by_cam=(),
    )
    tri = TriangulationResult(seq=0, skull_position_xyz=(1.0, 2.0, 3.0))
    out_k = kabsch.solve_with_context(packet, inference=inf, triangulated=tri)
    out_c = ceres.solve_with_context(packet, inference=inf, triangulated=tri)
    rk = Rotation.from_quat(
        [
            out_k.skull_quaternion_wxyz[1],
            out_k.skull_quaternion_wxyz[2],
            out_k.skull_quaternion_wxyz[3],
            out_k.skull_quaternion_wxyz[0],
        ]
    )
    rc = Rotation.from_quat(
        [
            out_c.skull_quaternion_wxyz[1],
            out_c.skull_quaternion_wxyz[2],
            out_c.skull_quaternion_wxyz[3],
            out_c.skull_quaternion_wxyz[0],
        ]
    )
    ang = (rk.inv() * rc).magnitude()
    assert ang < 0.02


def test_ceres_process_live_mocap_tick() -> None:
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    r_true = Rotation.from_euler("z", 15.0, degrees=True).as_matrix()
    obs = ref @ r_true.T
    keypoints_xyz = tuple((float(x), float(y), float(z)) for x, y, z in obs)

    class _Inf(RealtimeInferenceRuntime):
        def infer(self, packet, *, frame_set=None):
            return FrameInferenceResult(
                seq=packet.seq,
                confidence=1.0,
                keypoints_xyz=keypoints_xyz,
                single_landmark_uv_by_cam=(),
            )

    fs = LiveMocapFrameSet(seq=0, anchor_utc_ns=1_000, images_bgr={0: np.zeros((2, 2, 3), dtype=np.uint8)})
    solver = create_skull_solver("ceres", ceres_window_size=1)
    assert solver is not None
    fused = process_live_mocap_tick(
        fs,
        inference_runtime=_Inf(),
        triangulator=StubTriangulator(),
        calibrator=StubRollingEyeCalibrator(),
        fuser=StubGazeFuser(),
        skull_solver=solver,
    )
    r_exp = Rotation.from_matrix(r_true)
    r_out = Rotation.from_quat(
        [
            fused.skull_quaternion_wxyz[1],
            fused.skull_quaternion_wxyz[2],
            fused.skull_quaternion_wxyz[3],
            fused.skull_quaternion_wxyz[0],
        ]
    )
    assert (r_exp.inv() * r_out).magnitude() < 0.06


def test_ceres_sliding_window_averages_jitter() -> None:
    """Wide window pulls orientation toward shared mean vs single noisy frame."""
    ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
    r0 = Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
    obs_clean = ref @ r0.T
    rng = np.random.default_rng(42)
    obs_noisy = obs_clean + rng.normal(scale=0.8, size=obs_clean.shape)
    ceres_one = CeresRealtimeSkullSolver(ref.copy(), window_size=1, max_nfev=300)
    ceres_win = CeresRealtimeSkullSolver(ref.copy(), window_size=8, max_nfev=300)
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
    clean_inf = FrameInferenceResult(
        seq=0,
        confidence=1.0,
        keypoints_xyz=tuple((float(x), float(y), float(z)) for x, y, z in obs_clean),
        single_landmark_uv_by_cam=(),
    )
    for _ in range(7):
        ceres_win.solve_with_context(packet, inference=clean_inf, triangulated=tri)
    noisy_inf = FrameInferenceResult(
        seq=0,
        confidence=1.0,
        keypoints_xyz=tuple((float(x), float(y), float(z)) for x, y, z in obs_noisy),
        single_landmark_uv_by_cam=(),
    )
    out_one = ceres_one.solve_with_context(packet, inference=noisy_inf, triangulated=tri)
    out_win = ceres_win.solve_with_context(packet, inference=noisy_inf, triangulated=tri)
    r_true_rot = Rotation.from_matrix(r0)
    q1 = Rotation.from_quat(
        [
            out_one.skull_quaternion_wxyz[1],
            out_one.skull_quaternion_wxyz[2],
            out_one.skull_quaternion_wxyz[3],
            out_one.skull_quaternion_wxyz[0],
        ]
    )
    qw = Rotation.from_quat(
        [
            out_win.skull_quaternion_wxyz[1],
            out_win.skull_quaternion_wxyz[2],
            out_win.skull_quaternion_wxyz[3],
            out_win.skull_quaternion_wxyz[0],
        ]
    )
    e1 = (r_true_rot.inv() * q1).magnitude()
    ew = (r_true_rot.inv() * qw).magnitude()
    assert ew < e1


def test_ceres_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window_size"):
        CeresRealtimeSkullSolver(DEFAULT_KABSCH_REFERENCE_BODY.copy(), window_size=0)
