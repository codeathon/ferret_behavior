"""
Optional queue-driven ingest of Pupil eye frames (wall UTC) into :class:`PupilDualEyeRings`.

Use this from a dedicated thread or after decoding from ZMQ / IPC so the Basler
combiner thread stays light. Queue items are ``(eye_id, wall_utc_ns, bgr_uint8)``;
send ``None`` as a sentinel to stop the loop.
"""

from __future__ import annotations

import queue
import threading
from typing import Literal

import numpy as np

from src.cameras.synchronization.pupil_dual_eye_rings import PupilDualEyeRings
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

EyeId = Literal[0, 1]


def ingest_one_wall_utc_item(
    rings: PupilDualEyeRings,
    item: tuple[EyeId, int, np.ndarray],
) -> None:
    """Push one ``(eye_id, wall_utc_ns, image)`` tuple into ``rings``."""
    eye_id, wall_utc_ns, image = item
    if eye_id == 0:
        rings.push_eye0_wall(wall_utc_ns, image)
    else:
        rings.push_eye1_wall(wall_utc_ns, image)


def run_pupil_wall_utc_queue_ingest_loop(
    rings: PupilDualEyeRings,
    q: queue.Queue[tuple[EyeId, int, np.ndarray] | None],
) -> None:
    """
    Block on ``q`` until ``None``; each item is ingested into ``rings``.

    Intended to run on a non-daemon worker thread you join at shutdown.
    """
    while True:
        item = q.get()
        if item is None:
            break
        try:
            ingest_one_wall_utc_item(rings, item)
        except Exception as exc:
            logger.warning("Pupil queue ingest failed; dropping item: %s", exc)


class PupilWallUtcQueueIngestThread:
    """
    Background thread draining a queue into :class:`PupilDualEyeRings`.

    Why: decouple Pupil decode / network from Basler ``frameset_sink`` latency.
    """

    def __init__(self, rings: PupilDualEyeRings, max_queue: int = 256) -> None:
        self._rings = rings
        self._q: queue.Queue[tuple[EyeId, int, np.ndarray] | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    @property
    def queue(self) -> queue.Queue[tuple[EyeId, int, np.ndarray] | None]:
        return self._q

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("PupilWallUtcQueueIngestThread already started")
        self._thread = threading.Thread(
            target=run_pupil_wall_utc_queue_ingest_loop,
            args=(self._rings, self._q),
            name="pupil-wall-utc-queue-ingest",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 5.0) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            logger.warning("Pupil ingest queue full; cannot enqueue shutdown sentinel")
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            if self._thread.is_alive():
                logger.warning("Pupil ingest thread did not exit in time")
            self._thread = None
