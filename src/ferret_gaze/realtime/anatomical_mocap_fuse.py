"""
Anatomical live-mocap gaze fusion and binocular rolling calibration.

Places eye origins from skull-local rest offsets rotated into world, and
applies small per-eye yaw corrections in skull space learned from horizontal
disparity between the two gaze directions (slow EMA toward a target vergence).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.per_frame_compute import (
	FrameInferenceResult,
	RealtimeGazeFuser,
	RollingCalibrationState,
	RollingEyeCalibrator,
	StubGazeFuser,
	StubRollingEyeCalibrator,
	TriangulationResult,
)


def _normalize_vec3(v: np.ndarray) -> np.ndarray:
	"""Return a unit-length 3-vector or a zero vector if numerically degenerate."""
	n = float(np.linalg.norm(v))
	if n < 1e-12:
		return np.zeros(3, dtype=np.float64)
	return (v / n).astype(np.float64)


def _quat_wxyz_to_R_ws(w: float, x: float, y: float, z: float) -> np.ndarray:
	"""
	Skull basis in world: columns are skull +X,+Y,+Z expressed in world (right-handed).

	``v_world = R_ws @ v_skull`` for column vectors ``v_skull``.
	"""
	# SciPy expects quaternion in xyzw order.
	r = Rotation.from_quat(np.array([x, y, z, w], dtype=np.float64))
	return r.as_matrix()


def _yaw_xz_skull(d_skull: np.ndarray) -> float | None:
	"""Yaw angle (rad) in the skull XZ plane: atan2(x, z); None if undefined."""
	xz = float(np.hypot(d_skull[0], d_skull[2]))
	if xz < 1e-8:
		return None
	return float(np.arctan2(d_skull[0], d_skull[2]))


def _apply_yaw_y_skull(d_skull: np.ndarray, yaw_rad: float) -> np.ndarray:
	"""Rotate a skull-frame direction about local +Y by ``yaw_rad`` (column convention)."""
	ry = Rotation.from_euler("y", yaw_rad, degrees=False).as_matrix()
	return ry @ d_skull


class AnatomicalRollingEyeCalibrator(RollingEyeCalibrator):
	"""
	Binocular horizontal (yaw-in-XZ) EMA toward a target left-right yaw difference.

	Requires ``packet`` (skull quaternion + gaze dirs) and ``inference`` on the
	live path; falls back to the previous bias snapshot when either is missing.
	"""

	def __init__(
		self,
		*,
		vergence_ema_alpha: float = 0.08,
		target_left_minus_right_yaw_rad: float = 0.0,
		max_bias_rad: float = 0.45,
	) -> None:
		if not (0.0 < vergence_ema_alpha <= 1.0):
			raise ValueError("vergence_ema_alpha must be in (0, 1]")
		self._alpha = float(vergence_ema_alpha)
		self._target = float(target_left_minus_right_yaw_rad)
		self._max_b = float(max_bias_rad)
		self._lb = 0.0
		self._rb = 0.0

	def update(
		self,
		triangulated: TriangulationResult,
		*,
		inference: FrameInferenceResult | None = None,
		packet: RealtimeGazePacket | None = None,
	) -> RollingCalibrationState:
		# ``triangulated`` keeps signature parity with the stub calibrator (unused here).
		_ = triangulated
		gain = 1.0
		offset = 0.0
		if packet is None or inference is None:
			return RollingCalibrationState(
				gain=gain,
				offset=offset,
				left_yaw_bias_rad=self._lb,
				right_yaw_bias_rad=self._rb,
			)
		w, x, y, z = packet.skull_quaternion_wxyz
		r_ws = _quat_wxyz_to_R_ws(w, x, y, z)
		lw = _normalize_vec3(np.array(packet.left_gaze_direction_xyz, dtype=np.float64))
		rw = _normalize_vec3(np.array(packet.right_gaze_direction_xyz, dtype=np.float64))
		ls = r_ws.T @ lw
		rs = r_ws.T @ rw
		yaw_l = _yaw_xz_skull(ls)
		yaw_r = _yaw_xz_skull(rs)
		if yaw_l is not None and yaw_r is not None:
			err = (yaw_l - yaw_r) - self._target
			self._lb += self._alpha * (-0.5 * err)
			self._rb += self._alpha * (+0.5 * err)
			self._lb = float(np.clip(self._lb, -self._max_b, self._max_b))
			self._rb = float(np.clip(self._rb, -self._max_b, self._max_b))
		return RollingCalibrationState(
			gain=gain,
			offset=offset,
			left_yaw_bias_rad=self._lb,
			right_yaw_bias_rad=self._rb,
		)


class AnatomicalMocapGazeFuser(RealtimeGazeFuser):
	"""
	Fuse triangulated skull pose with skull-local eye geometry and calibrated gaze.

	Skull position comes from triangulation; quaternion from the incoming packet
	(Kabsch or scaffold). Eye rest positions are fixed in skull space then rotated
	to world. Gaze directions are expressed in skull space, yaw-corrected per
	eye, renormalized, and mapped back to world.
	"""

	def __init__(
		self,
		*,
		half_ipd_mm: float = 10.0,
		eye_y_mm: float = 25.0,
		eye_z_mm: float = 0.0,
	) -> None:
		if half_ipd_mm <= 0.0:
			raise ValueError("half_ipd_mm must be positive")
		self._left_skull = np.array([-half_ipd_mm, eye_y_mm, eye_z_mm], dtype=np.float64)
		self._right_skull = np.array([half_ipd_mm, eye_y_mm, eye_z_mm], dtype=np.float64)

	def fuse(
		self,
		packet: RealtimeGazePacket,
		triangulated: TriangulationResult,
		calibration: RollingCalibrationState,
		inference: FrameInferenceResult,
	) -> RealtimeGazePacket:
		x, y, z = triangulated.skull_position_xyz
		p_skull = np.array([x, y, z], dtype=np.float64)
		w, qx, qy, qz = packet.skull_quaternion_wxyz
		r_ws = _quat_wxyz_to_R_ws(w, qx, qy, qz)
		left_w = p_skull + r_ws @ self._left_skull
		right_w = p_skull + r_ws @ self._right_skull
		lw = _normalize_vec3(np.array(packet.left_gaze_direction_xyz, dtype=np.float64))
		rw = _normalize_vec3(np.array(packet.right_gaze_direction_xyz, dtype=np.float64))
		ls = r_ws.T @ lw
		rs = r_ws.T @ rw
		ls_c = _normalize_vec3(_apply_yaw_y_skull(ls, -float(calibration.left_yaw_bias_rad)))
		rs_c = _normalize_vec3(_apply_yaw_y_skull(rs, -float(calibration.right_yaw_bias_rad)))
		lw_out = _normalize_vec3(r_ws @ ls_c)
		rw_out = _normalize_vec3(r_ws @ rs_c)
		return packet.model_copy(
			update={
				"skull_position_xyz": (float(x), float(y), float(z)),
				"left_eye_origin_xyz": (float(left_w[0]), float(left_w[1]), float(left_w[2])),
				"right_eye_origin_xyz": (float(right_w[0]), float(right_w[1]), float(right_w[2])),
				"left_gaze_direction_xyz": (float(lw_out[0]), float(lw_out[1]), float(lw_out[2])),
				"right_gaze_direction_xyz": (float(rw_out[0]), float(rw_out[1]), float(rw_out[2])),
				"confidence": inference.confidence,
			},
		)


def create_eye_calibrator(
	backend: Literal["stub", "anatomical"] | str,
	*,
	vergence_ema_alpha: float = 0.08,
	target_left_minus_right_yaw_rad: float = 0.0,
	max_bias_rad: float = 0.45,
) -> RollingEyeCalibrator:
	"""Factory for rolling eye calibrators (stub stays default for legacy runs)."""
	b = str(backend).strip().lower()
	if b in ("", "stub"):
		return StubRollingEyeCalibrator()
	if b == "anatomical":
		return AnatomicalRollingEyeCalibrator(
			vergence_ema_alpha=vergence_ema_alpha,
			target_left_minus_right_yaw_rad=target_left_minus_right_yaw_rad,
			max_bias_rad=max_bias_rad,
		)
	raise ValueError(f"Unsupported eye_calibrator_backend: {backend!r}")


def create_gaze_fuser(
	backend: Literal["stub", "anatomical"] | str,
	*,
	half_ipd_mm: float = 10.0,
	eye_y_mm: float = 25.0,
	eye_z_mm: float = 0.0,
) -> RealtimeGazeFuser:
	"""Factory for gaze fusion backends."""
	b = str(backend).strip().lower()
	if b in ("", "stub"):
		return StubGazeFuser()
	if b == "anatomical":
		return AnatomicalMocapGazeFuser(
			half_ipd_mm=half_ipd_mm,
			eye_y_mm=eye_y_mm,
			eye_z_mm=eye_z_mm,
		)
	raise ValueError(f"Unsupported gaze_fuser_backend: {backend!r}")
