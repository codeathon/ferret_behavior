"""
Realtime synchronization primitives for live camera pipelines.

Step 5 scaffold:
- per-camera ring buffers for Basler frames
- in-memory frame-set combiner for near-synchronous multi-camera bundles
- nearest-frame association helpers for Pupil/Basler matching
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaslerFrame:
    """One captured Basler frame with UTC timestamp."""

    camera_id: int
    frame_index: int
    capture_utc_ns: int
    payload: Any | None = None


@dataclass(frozen=True)
class BaslerFrameSet:
    """One synchronized multi-camera frame-set."""

    anchor_utc_ns: int
    frames_by_camera: dict[int, BaslerFrame]


class BaslerRingBuffer:
    """Fixed-capacity ring buffer for one camera stream."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._frames: deque[BaslerFrame] = deque(maxlen=maxlen)

    def push(self, frame: BaslerFrame) -> None:
        """Append one frame, evicting oldest if full."""
        self._frames.append(frame)

    def nearest(self, target_utc_ns: int) -> BaslerFrame | None:
        """Return nearest frame to target timestamp."""
        if not self._frames:
            return None
        return min(self._frames, key=lambda frame: abs(frame.capture_utc_ns - target_utc_ns))

    def __len__(self) -> int:
        return len(self._frames)


class BaslerFrameSetCombiner:
    """
    Combine camera-local streams into near-synchronous frame-sets in memory.

    The combiner is intentionally lightweight: it selects nearest frames across
    camera ring buffers around the latest ingested frame timestamp.
    """

    def __init__(self, camera_ids: list[int], ring_size: int = 240, tolerance_ns: int = 2_000_000) -> None:
        if not camera_ids:
            raise ValueError("camera_ids cannot be empty")
        if tolerance_ns <= 0:
            raise ValueError("tolerance_ns must be positive")
        self._camera_ids = sorted(set(camera_ids))
        self._tolerance_ns = tolerance_ns
        self._buffers = {camera_id: BaslerRingBuffer(maxlen=ring_size) for camera_id in self._camera_ids}

    def ingest(self, frame: BaslerFrame) -> BaslerFrameSet | None:
        """Insert one frame and emit a synchronized frame-set if available."""
        if frame.camera_id not in self._buffers:
            raise ValueError(f"Unknown camera_id: {frame.camera_id}")
        self._buffers[frame.camera_id].push(frame)
        if any(len(buffer) == 0 for buffer in self._buffers.values()):
            return None

        anchor_utc_ns = frame.capture_utc_ns
        chosen: dict[int, BaslerFrame] = {}
        for camera_id, buffer in self._buffers.items():
            nearest = buffer.nearest(anchor_utc_ns)
            if nearest is None:
                return None
            if abs(nearest.capture_utc_ns - anchor_utc_ns) > self._tolerance_ns:
                return None
            chosen[camera_id] = nearest
        return BaslerFrameSet(anchor_utc_ns=anchor_utc_ns, frames_by_camera=chosen)


def nearest_timestamp_index(timestamps_utc_ns: list[int], target_utc_ns: int) -> int:
    """Return index of nearest timestamp in sorted or unsorted list."""
    if not timestamps_utc_ns:
        raise ValueError("timestamps_utc_ns cannot be empty")
    return min(range(len(timestamps_utc_ns)), key=lambda i: abs(timestamps_utc_ns[i] - target_utc_ns))


def associate_pupil_frames_to_basler(
    basler_utc_ns: int,
    pupil_eye0_timestamps_utc_ns: list[int],
    pupil_eye1_timestamps_utc_ns: list[int],
) -> tuple[int, int]:
    """Associate nearest Pupil eye frame indices for one Basler UTC timestamp."""
    eye0_idx = nearest_timestamp_index(pupil_eye0_timestamps_utc_ns, basler_utc_ns)
    eye1_idx = nearest_timestamp_index(pupil_eye1_timestamps_utc_ns, basler_utc_ns)
    return eye0_idx, eye1_idx
