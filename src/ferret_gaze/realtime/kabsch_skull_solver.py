"""
Kabsch rigid alignment for realtime skull orientation from matched 3D keypoints.

When inference provides at least three non-degenerate ``keypoints_xyz`` rows
(and the same count as the loaded reference template), we estimate rotation
from reference frame to observation and write ``skull_quaternion_wxyz`` on the
packet. Position is still supplied by triangulation + :class:`StubGazeFuser`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.per_frame_compute import FrameInferenceResult, TriangulationResult
from src.ferret_gaze.realtime.solver_benchmark import RealtimeSkullSolver

# Default template (skull-local-ish units); replace via ``.npy`` Nx3 float64 path.
DEFAULT_KABSCH_REFERENCE_BODY = np.array(
	[
		[0.0, 0.0, 0.0],
		[12.0, 0.0, 0.0],
		[0.0, 10.0, 0.0],
	],
	dtype=np.float64,
)


def kabsch_rotation_translation(observed: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	Least-squares rigid transform mapping ``reference`` rows toward ``observed`` rows.

	Both arrays are ``(N, 3)`` with identical ``N >= 1``. Returns ``R`` (3, 3) with
	``det(R) = +1`` and translation row-vector ``t`` (3,) such that
	``observed ≈ reference @ R.T + t`` (row-wise).
	"""
	if observed.shape != reference.shape or observed.ndim != 2 or observed.shape[1] != 3:
		raise ValueError("observed and reference must be the same (N, 3) shape")
	n = observed.shape[0]
	if n < 1:
		raise ValueError("Need at least one point")
	mu_o = observed.mean(axis=0)
	mu_r = reference.mean(axis=0)
	oc = observed - mu_o
	rc = reference - mu_r
	# Cross-covariance of row-centered coordinates: ``H = Oc.T @ Rc`` (3x3).
	# For ``observed ≈ reference @ R.T + t`` (row points), the proper rotation is ``U @ Vh``.
	h = oc.T @ rc
	u, _, vt = np.linalg.svd(h)
	r = u @ vt
	if np.linalg.det(r) < 0.0:
		vt[-1, :] *= -1.0
		r = u @ vt
	t = mu_o - mu_r @ r.T
	return r, t


def rotation_matrix_to_quaternion_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
	"""Convert a proper rotation matrix to unit quaternion ``(w, x, y, z)``."""
	quat_xyzw = Rotation.from_matrix(r).as_quat()
	x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
	norm = (w * w + x * x + y * y + z * z) ** 0.5
	if norm < 1e-12:
		return (1.0, 0.0, 0.0, 0.0)
	return (w / norm, x / norm, y / norm, z / norm)


def load_kabsch_reference_body(path: Path) -> np.ndarray:
	"""Load ``Nx3`` float64 reference points from ``.npy``."""
	if not path.is_file():
		raise FileNotFoundError(f"Kabsch reference .npy not found: {path}")
	arr = np.load(path, allow_pickle=False)
	arr = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
	if arr.shape[0] < 3:
		raise ValueError(f"Reference must have at least 3 rows, got {arr.shape[0]}")
	return arr


class KabschRealtimeSkullSolver(RealtimeSkullSolver):
	"""
	Estimate skull quaternion from paired reference vs observed keypoint triplets.

	``solve`` is a no-op for replay benchmarks that only pass packets. Live ticks
	should call :meth:`solve_with_context` (see :func:`process_live_mocap_tick`).
	"""

	def __init__(self, reference_body: np.ndarray) -> None:
		ref = np.asarray(reference_body, dtype=np.float64).reshape(-1, 3)
		if ref.shape[0] < 3:
			raise ValueError("Kabsch reference_body needs at least 3 rows")
		self._reference = ref

	@property
	def name(self) -> str:
		return "kabsch"

	def solve(self, packet: RealtimeGazePacket) -> RealtimeGazePacket:
		"""Replay-only path: do not alter packets without keypoint context."""
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
		try:
			r_mat, _ = kabsch_rotation_translation(p_obs, q_ref)
			wxyz = rotation_matrix_to_quaternion_wxyz(r_mat)
		except (np.linalg.LinAlgError, ValueError):
			return packet
		tx, ty, tz = triangulated.skull_position_xyz
		return packet.model_copy(
			update={
				"skull_position_xyz": (tx, ty, tz),
				"skull_quaternion_wxyz": wxyz,
			},
		)


def create_skull_solver(
	backend: Literal["none", "kabsch"] | str,
	*,
	kabsch_reference_npy: Path | None = None,
) -> RealtimeSkullSolver | None:
	"""
	Factory for optional skull solvers used on the live mocap path.

	``kabsch`` loads ``kabsch_reference_npy`` when set; otherwise uses
	``DEFAULT_KABSCH_REFERENCE_BODY``.
	"""
	b = str(backend).strip().lower()
	if b in ("", "none"):
		return None
	if b == "kabsch":
		ref = load_kabsch_reference_body(kabsch_reference_npy) if kabsch_reference_npy is not None else DEFAULT_KABSCH_REFERENCE_BODY.copy()
		return KabschRealtimeSkullSolver(reference_body=ref)
	raise ValueError(f"Unsupported skull_solver_backend: {backend!r}")
