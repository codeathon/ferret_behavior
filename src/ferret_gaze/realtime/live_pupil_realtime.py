"""
Live Pupil + Basler realtime session context (used by ``full_pipeline`` live_mocap grab).

Owns :class:`~src.cameras.synchronization.pupil_dual_eye_rings.PupilDualEyeRings`,
optional :class:`~src.cameras.synchronization.pupil_live_queue_ingest.PupilWallUtcQueueIngestThread`,
and installs :class:`~src.cameras.synchronization.utc_clock_bridge.WallUtcFromPupilTime` on the
Basler grab latch when a Pupil Remote endpoint is configured.
"""

from __future__ import annotations

import threading

from src.cameras.diagnostics.timestamp_mapping import TimestampMapping
from src.cameras.synchronization.pupil_clock_sync import PupilClockMapper
from src.cameras.synchronization.pupil_dual_eye_rings import PupilDualEyeRings
from src.cameras.synchronization.pupil_live_queue_ingest import PupilWallUtcQueueIngestThread
from src.cameras.synchronization.utc_clock_bridge import WallUtcFromPupilTime
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


class LiveMocapPupilRealtimeContext:
	"""
	Lifecycle for Pupil rings + optional wall-bridge on Basler latch + queue ingest.

	Constructed when realtime config enables live Pupil association; call
	:meth:`on_grab_latch` from the grab thread (via ``GrabLoopRunner``) so the bridge
	uses the same ``TimestampMapping`` as Basler ``capture_utc_ns``.
	"""

	def __init__(
		self,
		*,
		ring_maxlen: int,
		clock_sync_endpoint: str | None = None,
		start_queue_thread: bool = False,
	) -> None:
		self.rings = PupilDualEyeRings(maxlen_per_eye=ring_maxlen)
		self._clock_endpoint = clock_sync_endpoint
		self._bridge_lock = threading.Lock()
		self._bridge: WallUtcFromPupilTime | None = None
		self._queue_thread: PupilWallUtcQueueIngestThread | None = None
		if start_queue_thread:
			self._queue_thread = PupilWallUtcQueueIngestThread(self.rings)
			self._queue_thread.start()

	def on_grab_latch(self, mapping: TimestampMapping) -> None:
		"""Wire Pupil clock to Basler latch (optional); safe to call from grab thread."""
		if not self._clock_endpoint:
			return
		try:
			mapper = PupilClockMapper.from_live_samples(endpoint=self._clock_endpoint)
		except Exception as exc:
			logger.warning("Pupil clock sync at latch failed; pupil_time mapping disabled: %s", exc)
			return
		with self._bridge_lock:
			self._bridge = WallUtcFromPupilTime(latch=mapping, mapper=mapper)
		logger.info("Live Pupil wall-clock bridge installed at Basler latch")

	def wall_bridge(self) -> WallUtcFromPupilTime | None:
		"""Return bridge after latch (for threads pushing ``pupil_time``-stamped frames)."""
		with self._bridge_lock:
			return self._bridge

	@property
	def pupil_queue(self):
		"""Queue for :class:`~src.cameras.synchronization.pupil_live_queue_ingest.PupilWallUtcQueueIngestThread` when enabled."""
		if self._queue_thread is None:
			return None
		return self._queue_thread.queue

	def shutdown(self) -> None:
		"""Stop queue ingest thread if started."""
		if self._queue_thread is not None:
			self._queue_thread.stop()
			self._queue_thread = None
