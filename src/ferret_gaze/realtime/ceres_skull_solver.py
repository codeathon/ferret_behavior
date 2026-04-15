"""
Sliding-window SE(3) skull alignment using nonlinear least squares (SciPy).

This is a Ceres-*style* bundle step on matched 3D keypoints: we minimize
weighted point-to-point residuals in the same row convention as
:class:`~src.ferret_gaze.realtime.kabsch_skull_solver.KabschRealtimeSkullSolver`.
For ``window_size == 1`` the optimum matches the closed-form Kabsch solution
(up to numerical tolerance). Larger windows temporally stabilize noisy tracks.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.kabsch_skull_solver import (
	kabsch_rotation_translation,
	rotation_matrix_to_quaternion_wxyz,
)
from src.ferret_gaze.realtime.per_frame_compute import FrameInferenceResult, TriangulationResult
from src.ferret_gaze.realtime.solver_benchmark import RealtimeSkullSolver


def _se3_residuals(
	p: np.ndarray,
	ref: np.ndarray,
	obs_stack: list[np.ndarray],
	weights: np.ndarray,
) -> np.ndarray:
	"""Stacked sqrt-weighted residuals for ``pred_row = ref_row @ R.T + t`` minus ``obs``."""
	r_mat = Rotation.from_rotvec(p[:3]).as_matrix()
	t = p[3:6]
	pred = ref @ r_mat.T + t
	parts: list[np.ndarray] = []
	for w, ob in zip(weights, obs_stack, strict=True):
		parts.append((np.sqrt(float(w)) * (pred - ob)).ravel())
	return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _initial_guess_se3(ref: np.ndarray, obs: np.ndarray) -> np.ndarray:
	"""Kabsch-derived (rotvec, translation) for a single observation set."""
	r_mat, t_row = kabsch_rotation_translation(obs, ref)
	rv = Rotation.from_matrix(r_mat).as_rotvec()
	return np.concatenate([rv.astype(np.float64), t_row.astype(np.float64)])


def _optimize_se3(
	ref: np.ndarray,
	obs_stack: list[np.ndarray],
	weights: np.ndarray,
	*,
	soft_l1_f_scale: float | None,
	max_nfev: int,
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Return ``(R, t_row)`` or ``None`` if optimization fails."""
	if not obs_stack:
		return None
	p0 = _initial_guess_se3(ref, obs_stack[-1])
	ls_kwargs: dict = {
		"fun": lambda p, r=ref, o=obs_stack, w=weights: _se3_residuals(p, r, o, w),
		"x0": p0,
		"method": "trf",
		"max_nfev": int(max_nfev),
	}
	if soft_l1_f_scale is not None and float(soft_l1_f_scale) > 0.0:
		ls_kwargs["loss"] = "soft_l1"
		ls_kwargs["f_scale"] = float(soft_l1_f_scale)
	try:
		sol = least_squares(**ls_kwargs)
	except ValueError:
		return None
	p = sol.x
	if not np.all(np.isfinite(p)):
		return None
	r_mat = Rotation.from_rotvec(p[:3]).as_matrix()
	return r_mat, p[3:6].astype(np.float64)


class CeresRealtimeSkullSolver(RealtimeSkullSolver):
	"""
	Nonlinear SE(3) fit of a fixed reference template to recent observations.

	``solve`` is a no-op for replay benchmarks; live ticks use :meth:`solve_with_context`.
	"""

	def __init__(
		self,
		reference_body: np.ndarray,
		*,
		window_size: int = 1,
		soft_l1_f_scale: float | None = None,
		max_nfev: int = 100,
	) -> None:
		ref = np.asarray(reference_body, dtype=np.float64).reshape(-1, 3)
		if ref.shape[0] < 3:
			raise ValueError("Ceres reference_body needs at least 3 rows")
		if window_size < 1:
			raise ValueError("window_size must be at least 1")
		self._reference = ref
		self._window_size = int(window_size)
		self._soft_l1_f_scale = float(soft_l1_f_scale) if soft_l1_f_scale is not None and float(soft_l1_f_scale) > 0.0 else None
		self._max_nfev = int(max_nfev)
		self._win: deque[np.ndarray] = deque(maxlen=self._window_size)
		self._last_n: int | None = None

	@property
	def name(self) -> str:
		return "ceres_sliding_scipy"

	def solve(self, packet: RealtimeGazePacket) -> RealtimeGazePacket:
		"""Replay-only path: no keypoints without live context."""
		return packet

	def solve_with_context(
		self,
		packet: RealtimeGazePacket,
		*,
		inference: FrameInferenceResult | None = None,
		triangulated: TriangulationResult | None = None,
	) -> RealtimeGazePacket:
		if inference is None or triangulated is None:
			return packet
		pts = np.asarray(inference.keypoints_xyz, dtype=np.float64).reshape(-1, 3)
		if pts.shape[0] < 3:
			return packet
		n = min(pts.shape[0], self._reference.shape[0])
		if n < 3:
			return packet
		p_obs = pts[:n]
		q_ref = self._reference[:n]
		if self._last_n is not None and n != self._last_n:
			self._win.clear()
		self._last_n = n
		self._win.append(p_obs.copy())
		obs_list = list(self._win)
		w = np.ones(len(obs_list), dtype=np.float64)
		w /= float(np.sum(w))
		opt = _optimize_se3(q_ref, obs_list, w, soft_l1_f_scale=self._soft_l1_f_scale, max_nfev=self._max_nfev)
		if opt is None:
			try:
				r_mat, _ = kabsch_rotation_translation(p_obs, q_ref)
				wxyz = rotation_matrix_to_quaternion_wxyz(r_mat)
			except (np.linalg.LinAlgError, ValueError):
				return packet
		else:
			r_mat, _t = opt
			wxyz = rotation_matrix_to_quaternion_wxyz(r_mat)
		tx, ty, tz = triangulated.skull_position_xyz
		return packet.model_copy(
			update={
				"skull_position_xyz": (tx, ty, tz),
				"skull_quaternion_wxyz": wxyz,
			},
		)
