"""
Load freemocap-style session calibration TOML and build OpenCV projection matrices.

Each ``cam_*`` block matches ``log_cameras`` / Rerun expectations: intrinsics
``matrix`` (3x3 K), ``world_position`` (camera center in world), and
``world_orientation`` (3x3 with columns = camera x,y,z axes expressed in world).

World-to-camera rotation ``R`` and translation ``t`` satisfy
``X_cam = R @ X_world + t`` with ``P = K @ [R|t]`` (3x4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import toml

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def discover_session_calibration_toml(session_root: Path) -> Path | None:
    """
    Return the first ``*camera_calibration.toml`` under ``session_root/calibration``.

    Matches ``RecordingFolder.calibration_toml_path`` glob semantics.
    """
    cal_dir = session_root / "calibration"
    if not cal_dir.is_dir():
        return None
    matches = sorted(cal_dir.glob("*camera_calibration.toml"))
    return matches[0] if matches else None


def _as_float_matrix3(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{label} must be 3x3, got shape {arr.shape}")
    return arr


def _as_float_vector3(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{label} must have length 3, got {arr.size}")
    return arr


def projection_matrix_from_cam_block(block: dict[str, Any]) -> np.ndarray:
    """
    Build a 3x4 projection matrix ``P = K @ [R | t]`` from one TOML camera table.

    ``world_orientation`` is camera-to-world: columns are camera basis vectors in
    world coordinates (same convention as ``log_cameras`` / Rerun ``mat3x3``).
    """
    if "matrix" not in block or "world_position" not in block or "world_orientation" not in block:
        raise KeyError("camera block must include matrix, world_position, world_orientation")
    k = _as_float_matrix3(block["matrix"], label="matrix")
    o_cw = _as_float_matrix3(block["world_orientation"], label="world_orientation")
    c_w = _as_float_vector3(block["world_position"], label="world_position")
    # Columns of O are camera axes in world; world coordinates of point in camera frame:
    # X_cam = O^T (X_world - C)
    r = o_cw.T
    t = -r @ c_w.reshape(3, 1)
    rt = np.hstack([r, t])
    p = k @ rt
    return p.astype(np.float64)


def _sorted_cam_keys(raw: dict[str, Any]) -> list[str]:
    keys = [k for k in raw if isinstance(k, str) and k.startswith("cam_")]
    if not keys:
        raise ValueError("calibration TOML has no cam_* tables")
    return sorted(keys, key=lambda name: int(name.split("_", 1)[1]))


@dataclass(frozen=True)
class SessionMultiViewCalibration:
    """Per-camera 3x4 OpenCV projection matrices and labels keyed by cam index."""

    projection_by_cam_index: dict[int, np.ndarray]
    camera_name_by_index: dict[int, str]

    def projection_matrix(self, cam_index: int) -> np.ndarray:
        """Return 3x4 ``P`` for ``cam_{index}``."""
        if cam_index not in self.projection_by_cam_index:
            raise KeyError(f"No calibration for camera index {cam_index}")
        return self.projection_by_cam_index[cam_index]


def load_session_multi_view_calibration(toml_path: Path) -> SessionMultiViewCalibration:
    """
    Parse a ``*camera_calibration.toml`` and build projection matrices per cam index.

    Camera index ``i`` corresponds to TOML key ``cam_i`` after sorting by suffix.
    """
    if not toml_path.is_file():
        raise FileNotFoundError(f"Calibration TOML not found: {toml_path}")
    raw = toml.load(toml_path)
    names: dict[int, str] = {}
    projections: dict[int, np.ndarray] = {}
    for key in _sorted_cam_keys(raw):
        block = raw[key]
        if not isinstance(block, dict):
            raise TypeError(f"{key} must be a table, got {type(block)}")
        idx = int(key.split("_", 1)[1])
        name = str(block.get("name", key))
        names[idx] = name
        projections[idx] = projection_matrix_from_cam_block(block)
        logger.debug("Loaded projection for %s (index=%d, name=%s)", key, idx, name)
    return SessionMultiViewCalibration(
        projection_by_cam_index=projections,
        camera_name_by_index=names,
    )
