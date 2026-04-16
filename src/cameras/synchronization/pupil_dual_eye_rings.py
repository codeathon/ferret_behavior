"""
Thread-safe ring buffers of Pupil eye frames keyed by wall UTC (nanoseconds).

Ingest pushes (utc_ns, image) per eye; association picks the nearest UTC to a
Basler anchor using binary search on a monotonic timestamp list.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass

import numpy as np

from src.cameras.synchronization.utc_clock_bridge import WallUtcFromPupilTime
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PupilAssociationMetrics:
    """Mutable counters for live Pupil/Basler association (diagnostics / tests)."""

    lookups: int = 0
    stale_eye0: int = 0
    stale_eye1: int = 0
    missing_eye0_buffer: int = 0
    missing_eye1_buffer: int = 0


class _SortedUtcImageRing:
    """Fixed-capacity monotonic UTC list + aligned images; nearest lookup O(log n)."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 4:
            raise ValueError("maxlen must be at least 4")
        self._maxlen = maxlen
        self._utc: list[int] = []
        self._img: list[np.ndarray] = []

    def push(self, utc_ns: int, image: np.ndarray) -> None:
        # Drop non-monotonic samples (USB reordering / clock noise) to keep bisect valid.
        if self._utc and utc_ns < self._utc[-1]:
            logger.debug(
                "Pupil ring: non-monotonic utc_ns=%s after %s; dropping frame",
                utc_ns,
                self._utc[-1],
            )
            return
        self._utc.append(utc_ns)
        self._img.append(image)
        while len(self._utc) > self._maxlen:
            self._utc.pop(0)
            self._img.pop(0)

    def nearest(self, target_utc_ns: int) -> tuple[np.ndarray | None, int | None, int]:
        """
        Return (image, chosen_utc_ns, abs_delta_ns).

        When empty, returns (None, None, 2**62).
        """
        if not self._utc:
            return None, None, 2**62
        i = bisect.bisect_left(self._utc, target_utc_ns)
        best_i = 0
        best_d = abs(self._utc[0] - target_utc_ns)
        if i > 0:
            d0 = abs(self._utc[i - 1] - target_utc_ns)
            if d0 < best_d:
                best_d = d0
                best_i = i - 1
        if i < len(self._utc):
            d1 = abs(self._utc[i] - target_utc_ns)
            if d1 < best_d:
                best_d = d1
                best_i = i
        return self._img[best_i], self._utc[best_i], best_d

    def __len__(self) -> int:
        return len(self._utc)


@dataclass(frozen=True)
class PupilEyeAssociation:
    """One eye's nearest frame to a Basler anchor, or stale/missing markers."""

    image: np.ndarray | None
    chosen_utc_ns: int | None
    delta_ns: int
    stale: bool


class PupilDualEyeRings:
    """
    Thread-safe dual-eye buffers for live association to Basler frame-sets.

    Push either wall-UTC-tagged frames (:meth:`push_eye0_wall` / :meth:`push_eye1_wall`)
    or Pupil-device-time frames plus a :class:`WallUtcFromPupilTime` bridge.
    """

    def __init__(self, maxlen_per_eye: int = 512, metrics: PupilAssociationMetrics | None = None) -> None:
        self._e0 = _SortedUtcImageRing(maxlen_per_eye)
        self._e1 = _SortedUtcImageRing(maxlen_per_eye)
        self._lock = threading.Lock()
        self._metrics = metrics or PupilAssociationMetrics()

    @property
    def metrics(self) -> PupilAssociationMetrics:
        return self._metrics

    def push_eye0_wall(self, wall_utc_ns: int, image: np.ndarray) -> None:
        """Ingest eye0 frame already on wall UTC (e.g. tests or pre-mapped pipeline)."""
        with self._lock:
            self._e0.push(wall_utc_ns, image)

    def push_eye1_wall(self, wall_utc_ns: int, image: np.ndarray) -> None:
        """Ingest eye1 frame already on wall UTC."""
        with self._lock:
            self._e1.push(wall_utc_ns, image)

    def push_eye0_pupil_time(
        self,
        pupil_time_ns: int,
        image: np.ndarray,
        bridge: WallUtcFromPupilTime,
    ) -> None:
        """Map Pupil clock time to wall UTC then store eye0."""
        wall = bridge.pupil_time_ns_to_wall_utc_ns(pupil_time_ns)
        self.push_eye0_wall(wall, image)

    def push_eye1_pupil_time(
        self,
        pupil_time_ns: int,
        image: np.ndarray,
        bridge: WallUtcFromPupilTime,
    ) -> None:
        """Map Pupil clock time to wall UTC then store eye1."""
        wall = bridge.pupil_time_ns_to_wall_utc_ns(pupil_time_ns)
        self.push_eye1_wall(wall, image)

    def associate(
        self,
        basler_anchor_utc_ns: int,
        stale_max_delta_ns: int,
    ) -> tuple[PupilEyeAssociation, PupilEyeAssociation]:
        """
        Nearest-neighbor match per eye to ``basler_anchor_utc_ns``.

        ``stale_max_delta_ns``: if best delta exceeds this, ``stale`` is True and
        image may still be returned for debugging.
        """
        with self._lock:
            self._metrics.lookups += 1
            img0, utc0, d0 = self._e0.nearest(basler_anchor_utc_ns)
            img1, utc1, d1 = self._e1.nearest(basler_anchor_utc_ns)
            if img0 is None:
                self._metrics.missing_eye0_buffer += 1
            if img1 is None:
                self._metrics.missing_eye1_buffer += 1
            stale0 = img0 is None or d0 > stale_max_delta_ns
            stale1 = img1 is None or d1 > stale_max_delta_ns
            if stale0:
                self._metrics.stale_eye0 += 1
            if stale1:
                self._metrics.stale_eye1 += 1
            return (
                PupilEyeAssociation(image=img0, chosen_utc_ns=utc0, delta_ns=d0, stale=stale0),
                PupilEyeAssociation(image=img1, chosen_utc_ns=utc1, delta_ns=d1, stale=stale1),
            )
