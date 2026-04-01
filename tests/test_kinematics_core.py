"""
Tests for python_code/kinematics_core/

This module tests the foundational kinematic data models used throughout the
pipeline. Coverage includes:

- Quaternion: construction, normalization, multiplication, rotation, SLERP,
  rotation-matrix round-trips, Euler conversion.
- resample_quaternions: boundary clamping, interpolation correctness, error cases.
- RigidBodyKinematics: shape validation, velocity from constant/linear position,
  resampling (up/down), timestamp shifting, save/load round-trip.
- ReferenceGeometry: JSON round-trip, keypoint validation, distance computation.
- CoordinateFrameDefinition: axis validation rules.
"""

import json
import math

import numpy as np
import pytest

from python_code.kinematics_core.quaternion_model import Quaternion, resample_quaternions
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.reference_geometry_model import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    MarkerPosition,
    ReferenceGeometry,
)
from python_code.kinematics_core.kinematics_serialization import save_kinematics, load_kinematics


# =============================================================================
# Quaternion
# =============================================================================

class TestQuaternion:
    def test_identity_has_unit_norm(self):
        q = Quaternion.identity()
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert abs(norm - 1.0) < 1e-9

    def test_unnormalized_input_is_normalized(self):
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        assert abs(q.w - 1.0) < 1e-9

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError):
            Quaternion(w=0.0, x=0.0, y=0.0, z=0.0)

    def test_conjugate(self):
        q = Quaternion(w=1.0, x=0.1, y=0.2, z=0.3)
        c = q.conjugate()
        assert c.w == pytest.approx(q.w)
        assert c.x == pytest.approx(-q.x)
        assert c.y == pytest.approx(-q.y)
        assert c.z == pytest.approx(-q.z)

    def test_multiply_identity_is_noop(self):
        q = Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)
        result = q * Quaternion.identity()
        assert result.w == pytest.approx(q.w, abs=1e-6)
        assert result.x == pytest.approx(q.x, abs=1e-6)

    def test_q_times_conjugate_is_identity(self):
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        result = q * q.conjugate()
        assert result.w == pytest.approx(1.0, abs=1e-6)
        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert abs(result.z) < 1e-6

    def test_rotate_vector_identity_unchanged(self):
        q = Quaternion.identity()
        v = np.array([1.0, 2.0, 3.0])
        result = q.rotate_vector(v)
        assert np.allclose(result, v, atol=1e-9)

    def test_rotate_vector_90_deg_around_z(self):
        angle = math.pi / 2
        q = Quaternion(w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2))
        v = np.array([1.0, 0.0, 0.0])
        result = q.rotate_vector(v)
        assert np.allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_rotation_matrix_round_trip(self):
        angle = 0.6
        q = Quaternion(w=math.cos(angle / 2), x=math.sin(angle / 2), y=0.0, z=0.0)
        R = q.to_rotation_matrix()
        q2 = Quaternion.from_rotation_matrix(R)
        # q and -q represent the same rotation
        dot = abs(q.w * q2.w + q.x * q2.x + q.y * q2.y + q.z * q2.z)
        assert dot == pytest.approx(1.0, abs=1e-6)

    def test_to_euler_identity_is_zero(self):
        roll, pitch, yaw = Quaternion.identity().to_euler_xyz()
        assert abs(roll) < 1e-9
        assert abs(pitch) < 1e-9
        assert abs(yaw) < 1e-9

    def test_slerp_at_t0_returns_q0(self):
        q0 = Quaternion.identity()
        angle = math.pi / 3
        q1 = Quaternion(w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2))
        result = Quaternion.slerp(q0, q1, t=0.0)
        assert result.w == pytest.approx(q0.w, abs=1e-6)

    def test_slerp_at_t1_returns_q1(self):
        q0 = Quaternion.identity()
        angle = math.pi / 3
        q1 = Quaternion(w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2))
        result = Quaternion.slerp(q0, q1, t=1.0)
        dot = abs(result.w * q1.w + result.x * q1.x + result.y * q1.y + result.z * q1.z)
        assert dot == pytest.approx(1.0, abs=1e-6)

    def test_slerp_midpoint_is_half_angle(self):
        angle = math.pi / 2
        q0 = Quaternion.identity()
        q1 = Quaternion(w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2))
        mid = Quaternion.slerp(q0, q1, t=0.5)
        _, mid_angle = mid.to_axis_angle()
        assert mid_angle == pytest.approx(angle / 2, abs=1e-5)

    def test_slerp_invalid_t_raises(self):
        q = Quaternion.identity()
        with pytest.raises(ValueError):
            Quaternion.slerp(q, q, t=1.5)


class TestResampleQuaternions:
    def _make_identity_list(self, n):
        return [Quaternion.identity() for _ in range(n)]

    def test_identity_resampled_stays_identity(self):
        quats = self._make_identity_list(10)
        ts = np.linspace(0.0, 1.0, 10)
        target = np.linspace(0.0, 1.0, 20)
        result = resample_quaternions(quats, ts, target)
        assert len(result) == 20
        for q in result:
            assert q.w == pytest.approx(1.0, abs=1e-6)

    def test_clamp_before_start(self):
        quats = self._make_identity_list(5)
        ts = np.linspace(1.0, 2.0, 5)
        target = np.array([0.0])  # before range
        result = resample_quaternions(quats, ts, target)
        assert result[0].w == pytest.approx(quats[0].w, abs=1e-6)

    def test_clamp_after_end(self):
        quats = self._make_identity_list(5)
        ts = np.linspace(0.0, 1.0, 5)
        target = np.array([2.0])  # after range
        result = resample_quaternions(quats, ts, target)
        assert result[0].w == pytest.approx(quats[-1].w, abs=1e-6)

    def test_mismatched_lengths_raise(self):
        quats = self._make_identity_list(5)
        with pytest.raises(ValueError):
            resample_quaternions(quats, np.linspace(0, 1, 4), np.linspace(0, 1, 10))

    def test_too_few_quaternions_raises(self):
        with pytest.raises(ValueError):
            resample_quaternions([Quaternion.identity()], np.array([0.0]), np.array([0.0]))


# =============================================================================
# RigidBodyKinematics
# =============================================================================

class TestRigidBodyKinematics:
    def test_construction_shape_mismatch_raises(self, minimal_reference_geometry):
        with pytest.raises(Exception):
            RigidBodyKinematics.from_pose_arrays(
                name="test",
                reference_geometry=minimal_reference_geometry,
                timestamps=np.linspace(0, 1, 10),
                position_xyz=np.zeros((5, 3)),   # wrong shape
                quaternions_wxyz=np.tile([1, 0, 0, 0], (10, 1)).astype(float),
            )

    def test_n_frames(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=50)
        assert kin.n_frames == 50

    def test_framerate(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=100, framerate_hz=30.0)
        assert kin.framerate_hz == pytest.approx(30.0, rel=1e-3)

    def test_constant_position_gives_zero_velocity(self, minimal_reference_geometry):
        n = 100
        timestamps = np.linspace(0, 1, n)
        position_xyz = np.zeros((n, 3))
        quats = np.zeros((n, 4))
        quats[:, 0] = 1.0
        kin = RigidBodyKinematics.from_pose_arrays(
            name="static",
            reference_geometry=minimal_reference_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quats,
        )
        assert np.allclose(kin.velocity_xyz[1:], 0.0, atol=1e-9)

    def test_linear_position_gives_constant_velocity(self, minimal_reference_geometry):
        n = 100
        dt = 0.01
        timestamps = np.arange(n) * dt
        position_xyz = np.zeros((n, 3))
        position_xyz[:, 0] = timestamps * 10.0  # vx = 10 mm/s
        quats = np.zeros((n, 4))
        quats[:, 0] = 1.0
        kin = RigidBodyKinematics.from_pose_arrays(
            name="linear",
            reference_geometry=minimal_reference_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quats,
        )
        assert np.allclose(kin.velocity_xyz[1:, 0], 10.0, atol=1e-6)

    def test_resample_changes_frame_count(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=100, framerate_hz=30.0)
        target = np.linspace(kin.timestamps[0], kin.timestamps[-1], 50)
        resampled = kin.resample(target)
        assert resampled.n_frames == 50

    def test_resample_preserves_start_end_position(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=100)
        target = np.linspace(kin.timestamps[0], kin.timestamps[-1], 200)
        resampled = kin.resample(target)
        assert resampled.position_xyz[0] == pytest.approx(kin.position_xyz[0], abs=1e-6)
        assert resampled.position_xyz[-1] == pytest.approx(kin.position_xyz[-1], abs=1e-6)

    def test_shift_timestamps(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=50)
        shifted = kin.shift_timestamps(-kin.timestamps[0])
        assert shifted.timestamps[0] == pytest.approx(0.0, abs=1e-9)
        assert shifted.n_frames == kin.n_frames

    def test_save_load_round_trip(self, make_rigid_body_kinematics, tmp_path):
        kin = make_rigid_body_kinematics(n_frames=20, name="skull")
        save_kinematics(kin, tmp_path)
        loaded = load_kinematics(
            reference_geometry_path=tmp_path / "skull_reference_geometry.json",
            kinematics_csv_path=tmp_path / "skull_kinematics.csv",
        )
        assert loaded.n_frames == kin.n_frames
        assert np.allclose(loaded.timestamps, kin.timestamps, atol=1e-9)
        assert np.allclose(loaded.position_xyz, kin.position_xyz, atol=1e-6)
        assert np.allclose(
            np.abs(loaded.quaternions_wxyz), np.abs(kin.quaternions_wxyz), atol=1e-6
        )

    def test_get_quaternion_at_frame(self, make_rigid_body_kinematics):
        kin = make_rigid_body_kinematics(n_frames=10)
        q = kin.get_quaternion(0)
        assert isinstance(q, Quaternion)
        assert q.w == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# ReferenceGeometry
# =============================================================================

class TestReferenceGeometry:
    def test_json_round_trip(self, minimal_reference_geometry, tmp_path):
        path = tmp_path / "ref_geom.json"
        minimal_reference_geometry.to_json_file(path)
        loaded = ReferenceGeometry.from_json_file(path)
        assert set(loaded.keypoints.keys()) == set(minimal_reference_geometry.keypoints.keys())
        for name in minimal_reference_geometry.keypoints:
            orig = minimal_reference_geometry.keypoints[name].to_array()
            reloaded = loaded.keypoints[name].to_array()
            assert np.allclose(orig, reloaded)

    def test_keypoint_local_positions_array_shape(self, minimal_reference_geometry):
        arr = minimal_reference_geometry.keypoint_local_positions_array
        assert arr.shape == (3, 3)  # 3 keypoints, 3 coords

    def test_distance_between_keypoints(self, minimal_reference_geometry):
        d = minimal_reference_geometry.distance_between_keypoints("left_eye", "right_eye")
        assert d == pytest.approx(24.0, abs=1e-6)

    def test_missing_keypoint_raises(self, minimal_reference_geometry):
        with pytest.raises(KeyError):
            minimal_reference_geometry.get_keypoint_position("nonexistent")

    def test_invalid_axis_reference_raises(self):
        with pytest.raises(Exception):
            ReferenceGeometry(
                units="mm",
                coordinate_frame=CoordinateFrameDefinition(
                    origin_keypoints=["a"],
                    x_axis=AxisDefinition(keypoints=["missing_marker"], type=AxisType.EXACT),
                    y_axis=AxisDefinition(keypoints=["a"], type=AxisType.APPROXIMATE),
                ),
                keypoints={"a": MarkerPosition(x=1.0, y=0.0, z=0.0)},
            )


class TestCoordinateFrameDefinition:
    def test_exactly_two_axes_required(self):
        with pytest.raises(Exception):
            CoordinateFrameDefinition(
                origin_keypoints=["a"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
            )

    def test_one_exact_one_approximate_required(self):
        with pytest.raises(Exception):
            CoordinateFrameDefinition(
                origin_keypoints=["a"],
                x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
                y_axis=AxisDefinition(keypoints=["b"], type=AxisType.EXACT),
            )

    def test_valid_definition_succeeds(self):
        frame = CoordinateFrameDefinition(
            origin_keypoints=["a"],
            x_axis=AxisDefinition(keypoints=["a"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["b"], type=AxisType.APPROXIMATE),
        )
        assert frame.x_axis is not None
        assert frame.y_axis is not None
