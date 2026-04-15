"""Tests for GrabLoopRunner -> live mocap queue wiring."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.cameras.synchronization.realtime_sync import BaslerFrame, BaslerFrameSet
from src.ferret_gaze.realtime.grab_live_wiring import LiveMocapGrabPublishWire
from src.ferret_gaze.realtime.per_frame_compute import StubInferenceRuntime, StubTriangulator
from src.ferret_gaze.realtime.publisher import NoOpRealtimePublisher


def test_grab_wire_sink_enqueues_live_bundle() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=8)
	img = np.zeros((3, 4, 3), dtype=np.uint8)
	bset = BaslerFrameSet(
		anchor_utc_ns=123,
		frames_by_camera={
			0: BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=123, payload=img),
		},
	)
	wire(bset)
	assert wire.pending_count() == 1


def test_grab_wire_sink_skips_empty_payload() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=8)
	bset = BaslerFrameSet(
		anchor_utc_ns=1,
		frames_by_camera={
			0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1, payload=None),
		},
	)
	wire(bset)
	assert wire.pending_count() == 0


def test_grab_wire_background_publisher_drains_queue() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=8)
	pub = NoOpRealtimePublisher()
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	img = np.zeros((2, 2, 3), dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=500,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=500, payload=img),
			},
		)
	)
	for _ in range(200):
		if wire.pending_count() == 0:
			break
		time.sleep(0.01)
	assert wire.pending_count() == 0
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 1


def test_grab_wire_double_start_raises() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=4)
	pub = NoOpRealtimePublisher()
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	with pytest.raises(RuntimeError, match="already running"):
		wire.start_background_publisher(
			publisher=pub,
			inference_runtime=StubInferenceRuntime(),
			triangulator=StubTriangulator(),
			stale_threshold_ms=80.0,
		)
	wire.stop_background_publisher()


def test_grab_wire_bounded_queue_drops_before_consumer_starts() -> None:
	"""With max depth 1, a second sink call evicts the first bundle before publish."""
	wire = LiveMocapGrabPublishWire(max_queue_size=1)
	img1 = np.full((2, 2, 3), 11, dtype=np.uint8)
	img2 = np.full((2, 2, 3), 99, dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=1,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1, payload=img1),
			},
		)
	)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=2,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=2, payload=img2),
			},
		)
	)
	assert wire.pending_count() == 1
	pub = NoOpRealtimePublisher()
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	for _ in range(200):
		if wire.pending_count() == 0:
			break
		time.sleep(0.01)
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 1


def test_grab_wire_drop_newest_policy_counts_overflow() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=1)
	wire.configure_overflow_policy(overflow_policy="drop_newest", put_timeout_ms=5)
	img = np.zeros((2, 2, 3), dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=1,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1, payload=img),
			},
		)
	)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=2,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=2, payload=img),
			},
		)
	)
	pub = NoOpRealtimePublisher()
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	for _ in range(200):
		if wire.pending_count() == 0:
			break
		time.sleep(0.01)
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 1
	assert summary.queue_overflow_count == 1


def test_grab_wire_block_with_timeout_policy_counts_overflow() -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=1)
	wire.configure_overflow_policy(overflow_policy="block_with_timeout", put_timeout_ms=1)
	img = np.zeros((2, 2, 3), dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=1,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1, payload=img),
			},
		)
	)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=2,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=2, payload=img),
			},
		)
	)
	pub = NoOpRealtimePublisher()
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	for _ in range(200):
		if wire.pending_count() == 0:
			break
		time.sleep(0.01)
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 1
	assert summary.queue_overflow_count == 1


def test_grab_wire_records_stage_errors(monkeypatch: pytest.MonkeyPatch) -> None:
	wire = LiveMocapGrabPublishWire(max_queue_size=8)
	pub = NoOpRealtimePublisher()
	monkeypatch.setattr(
		"src.ferret_gaze.realtime.grab_live_wiring.process_live_mocap_tick",
		lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stage boom")),
	)
	wire.start_background_publisher(
		publisher=pub,
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	img = np.zeros((2, 2, 3), dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=5,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=5, payload=img),
			},
		)
	)
	time.sleep(0.05)
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 0
	assert summary.stage_error_count == 1


def test_grab_wire_records_publish_errors() -> None:
	class _FailingPublisher(NoOpRealtimePublisher):
		def publish(self, packet):  # type: ignore[override]
			_ = packet
			raise RuntimeError("publish boom")

	wire = LiveMocapGrabPublishWire(max_queue_size=8)
	wire.start_background_publisher(
		publisher=_FailingPublisher(),
		inference_runtime=StubInferenceRuntime(),
		triangulator=StubTriangulator(),
		stale_threshold_ms=80.0,
	)
	img = np.zeros((2, 2, 3), dtype=np.uint8)
	wire(
		BaslerFrameSet(
			anchor_utc_ns=8,
			frames_by_camera={
				0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=8, payload=img),
			},
		)
	)
	time.sleep(0.05)
	summary = wire.stop_background_publisher(join_timeout_s=5.0)
	assert summary is not None
	assert summary.packet_count == 0
	assert summary.publish_error_count == 1
