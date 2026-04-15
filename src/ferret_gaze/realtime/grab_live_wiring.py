"""
Wire ``GrabLoopRunner`` combiner output into the live gaze compute path.

``GrabLoopRunner(..., frameset_sink=wire)`` invokes the sink from the combiner
thread; this module queues :class:`~src.ferret_gaze.realtime.live_frame_set.LiveMocapFrameSet`
and optionally runs a background publisher so the sink stays non-blocking.
"""

from __future__ import annotations

import queue
import threading
import time

from src.cameras.synchronization.realtime_sync import BaslerFrameSet
from src.ferret_gaze.realtime.latency_metrics import (
	LatencySummary,
	RealtimeLatencyMetrics,
	format_latency_summary,
)
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.live_mocap_pipeline import (
	basler_frameset_to_live_mocap_frame_set,
	process_live_mocap_tick,
)
from src.ferret_gaze.realtime.per_frame_compute import (
	RealtimeGazeFuser,
	RealtimeInferenceRuntime,
	RealtimeTriangulator,
	RollingEyeCalibrator,
	StubGazeFuser,
	StubRollingEyeCalibrator,
)
from src.ferret_gaze.realtime.publisher import RealtimePublisher
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

_SENTINEL = object()


class LiveMocapGrabPublishWire:
	"""
	Callable sink for ``GrabLoopRunner`` plus optional background publish loop.

	Usage::

		wire = LiveMocapGrabPublishWire(max_queue_size=32)
		wire.start_background_publisher(
		    publisher=publisher,
		    inference_runtime=infer_rt,
		    triangulator=tri,
		    stale_threshold_ms=80.0,
		)
		recording.grab_n_frames(n, frameset_sink=wire)
		wire.stop_background_publisher()
		publisher.close()

	The combiner thread calls ``wire(basler_frameset)``; keep work minimal (queue only).
	"""

	def __init__(self, max_queue_size: int = 32) -> None:
		if max_queue_size < 1:
			raise ValueError("max_queue_size must be at least 1")
		self._q: queue.Queue[LiveMocapFrameSet | object] = queue.Queue(maxsize=max_queue_size)
		self._seq = 0
		self._thread: threading.Thread | None = None
		self._last_summary: LatencySummary | None = None

	def __call__(self, basler: BaslerFrameSet) -> None:
		"""Enqueue one live bundle; drops oldest item if the queue is full."""
		live = basler_frameset_to_live_mocap_frame_set(basler, self._seq)
		self._seq += 1
		if live is None:
			return
		try:
			self._q.put_nowait(live)
		except queue.Full:
			try:
				self._q.get_nowait()
			except queue.Empty:
				pass
			try:
				self._q.put_nowait(live)
			except queue.Full:
				logger.warning("Live mocap grab wire queue full; dropping frame set")

	def pending_count(self) -> int:
		"""Approximate queued bundles (for tests / diagnostics)."""
		return self._q.qsize()

	def start_background_publisher(
		self,
		*,
		publisher: RealtimePublisher,
		inference_runtime: RealtimeInferenceRuntime,
		triangulator: RealtimeTriangulator,
		stale_threshold_ms: float,
		calibrator: RollingEyeCalibrator | None = None,
		fuser: RealtimeGazeFuser | None = None,
		pace_hz: float | None = None,
	) -> None:
		"""Start a daemon thread that drains the queue and publishes fused packets."""
		if self._thread is not None and self._thread.is_alive():
			raise RuntimeError("LiveMocapGrabPublishWire consumer thread is already running")
		self._last_summary = None
		calib = calibrator or StubRollingEyeCalibrator()
		fuse = fuser or StubGazeFuser()
		self._thread = threading.Thread(
			target=self._consumer_loop,
			name="live-mocap-grab-publish",
			daemon=True,
			kwargs={
				"publisher": publisher,
				"inference_runtime": inference_runtime,
				"triangulator": triangulator,
				"stale_threshold_ms": stale_threshold_ms,
				"calibrator": calib,
				"fuser": fuse,
				"pace_hz": pace_hz,
			},
		)
		self._thread.start()
		logger.info("Live mocap grab publish consumer thread started")

	def _consumer_loop(
		self,
		*,
		publisher: RealtimePublisher,
		inference_runtime: RealtimeInferenceRuntime,
		triangulator: RealtimeTriangulator,
		stale_threshold_ms: float,
		calibrator: RollingEyeCalibrator,
		fuser: RealtimeGazeFuser,
		pace_hz: float | None,
	) -> None:
		metrics = RealtimeLatencyMetrics(stale_threshold_ms=stale_threshold_ms)
		period_s = (1.0 / pace_hz) if pace_hz is not None and pace_hz > 0 else None
		while True:
			try:
				item = self._q.get(timeout=0.5)
			except queue.Empty:
				continue
			if item is _SENTINEL:
				break
			assert isinstance(item, LiveMocapFrameSet)
			t_loop = time.perf_counter()
			fused = process_live_mocap_tick(
				item,
				inference_runtime=inference_runtime,
				triangulator=triangulator,
				calibrator=calibrator,
				fuser=fuser,
			)
			fused = fused.model_copy(update={"publish_utc_ns": time.time_ns()})
			publisher.publish(fused)
			metrics.observe(packet=fused, now_utc_ns=time.time_ns())
			if period_s is not None:
				elapsed = time.perf_counter() - t_loop
				sleep_s = period_s - elapsed
				if sleep_s > 0:
					time.sleep(sleep_s)
		self._last_summary = metrics.summary()
		logger.info(format_latency_summary(self._last_summary))

	def stop_background_publisher(self, *, join_timeout_s: float = 10.0) -> LatencySummary | None:
		"""Signal shutdown, enqueue sentinel, and join the consumer thread."""
		try:
			self._q.put(_SENTINEL, block=True, timeout=60.0)
		except Exception as exc:
			logger.warning("Could not enqueue live mocap shutdown sentinel: %s", exc)
		if self._thread is not None:
			self._thread.join(timeout=join_timeout_s)
			if self._thread.is_alive():
				logger.warning("Live mocap grab publish consumer thread did not exit in time")
			self._thread = None
		return self._last_summary
