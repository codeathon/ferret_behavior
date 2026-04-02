"""
Adapters between project quaternion conventions and common math backends.

This module keeps kinematics_core data models backend-agnostic while allowing
optional use of SciPy, numpy-quaternion, and pytransform3d for computation.

Project quaternion convention: [w, x, y, z]
SciPy quaternion convention:   [x, y, z, w]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) from [w, x, y, z] to [x, y, z, w]."""
    array = np.asarray(wxyz, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("Expected last dimension size 4 for quaternion values")
    return np.concatenate([array[..., 1:], array[..., :1]], axis=-1)


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) from [x, y, z, w] to [w, x, y, z]."""
    array = np.asarray(xyzw, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("Expected last dimension size 4 for quaternion values")
    return np.concatenate([array[..., 3:], array[..., :3]], axis=-1)


def require_scipy_rotation() -> type["Rotation"]:
    """Return scipy Rotation class or raise a clear dependency error."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for Rotation backend helpers. "
            "Install with: uv add scipy"
        ) from exc
    return Rotation


def scipy_rotation_from_wxyz(wxyz: np.ndarray) -> "Rotation":
    """Build scipy Rotation from project-format quaternion(s) [w, x, y, z]."""
    rotation_cls = require_scipy_rotation()
    return rotation_cls.from_quat(wxyz_to_xyzw(wxyz))


def wxyz_from_scipy_rotation(rotation: "Rotation") -> np.ndarray:
    """Convert scipy Rotation object to project-format quaternion(s) [w, x, y, z]."""
    return xyzw_to_wxyz(rotation.as_quat())


def require_numpy_quaternion():
    """Return numpy-quaternion module or raise a clear dependency error."""
    try:
        import quaternion
    except ImportError as exc:
        raise ImportError(
            "numpy-quaternion is required for quaternion backend helpers. "
            "Install with: uv add numpy-quaternion"
        ) from exc
    return quaternion


def numpy_quaternion_from_wxyz(wxyz: np.ndarray):
    """Convert project-format quaternion array [w, x, y, z] to numpy-quaternion."""
    quaternion = require_numpy_quaternion()
    array = np.asarray(wxyz, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("Expected last dimension size 4 for quaternion values")
    return quaternion.as_quat_array(array)


def wxyz_from_numpy_quaternion(quat_array) -> np.ndarray:
    """Convert numpy-quaternion array back to project-format [w, x, y, z]."""
    quaternion = require_numpy_quaternion()
    return np.asarray(quaternion.as_float_array(quat_array), dtype=np.float64)


def require_pytransform3d():
    """Import pytransform3d transformation helpers or raise clear error."""
    try:
        from pytransform3d.rotations import matrix_from_quaternion
        from pytransform3d.transformations import transform_from
    except ImportError as exc:
        raise ImportError(
            "pytransform3d is required for transform helpers. "
            "Install with: uv add pytransform3d"
        ) from exc
    return matrix_from_quaternion, transform_from


def transform_from_pose_wxyz(position_xyz: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from project pose values.

    Args:
        position_xyz: [x, y, z]
        orientation_wxyz: [w, x, y, z]
    """
    matrix_from_quaternion, transform_from = require_pytransform3d()
    position = np.asarray(position_xyz, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position_xyz must be shape (3,)")
    xyzw = wxyz_to_xyzw(np.asarray(orientation_wxyz, dtype=np.float64))
    rotation_matrix = matrix_from_quaternion(xyzw)
    return transform_from(R=rotation_matrix, p=position)
