"""
Linear multi-view triangulation (DLT) for calibrated pinhole cameras.

Used by ``MultiviewOpenCvTriangulator`` when at least two views provide pixel
coordinates for the same world point. OpenCV's ``triangulatePoints`` is limited
to two views; this module stacks all view constraints in one SVD solve.
"""

from __future__ import annotations

import numpy as np


def triangulate_linear_dlt(
    projections: list[np.ndarray],
    uv_pixels: list[np.ndarray],
) -> np.ndarray:
    """
    Triangulate one 3D point from 2+ pinhole views.

    Args:
        projections: Each element is a 3x4 projection matrix ``P = K[R|t]``.
        uv_pixels: Matching pixel coordinates ``(u, v)`` per view (image coords).

    Returns:
        Inhomogeneous world coordinates ``(x, y, z)`` as shape ``(3,)``.
    """
    if len(projections) < 2 or len(projections) != len(uv_pixels):
        raise ValueError("Need at least two matching (P, uv) pairs")
    rows: list[np.ndarray] = []
    for p, uv in zip(projections, uv_pixels, strict=True):
        p = np.asarray(p, dtype=np.float64).reshape(3, 4)
        uv = np.asarray(uv, dtype=np.float64).reshape(2)
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * p[2, :] - p[0, :])
        rows.append(v * p[2, :] - p[1, :])
    a = np.vstack(rows)
    _, _, vt = np.linalg.svd(a)
    x_h = vt[-1, :4].astype(np.float64)
    if abs(x_h[3]) < 1e-12:
        raise ValueError("Triangulation failed: degenerate homogeneous depth")
    x_h = x_h / x_h[3]
    return x_h[:3]
