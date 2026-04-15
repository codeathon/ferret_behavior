"""
Live synchronized camera frame bundle for the realtime mocap path.

Single anchor time + per-camera images for one pipeline tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LiveMocapFrameSet:
    """One multi-camera instant: UTC anchor + BGR images keyed by camera index."""

    seq: int
    """Monotonic frame index within the live session."""

    anchor_utc_ns: int
    """Primary timeline instant for this bundle (UTC nanoseconds since Unix epoch)."""

    images_bgr: Mapping[int, np.ndarray]
    """camera_index -> HxWx3 uint8 BGR image."""

    camera_serial_by_index: Mapping[int, str] = field(default_factory=dict)
    """Optional Basler serial (or label) per camera index for diagnostics."""
