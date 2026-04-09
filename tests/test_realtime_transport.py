"""
Tests for realtime transport scaffold components.

These tests cover schema validation and backend-agnostic transport interfaces
without requiring camera hardware, solver output, or ZMQ.
"""

import pytest

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.latency_metrics import (
    RealtimeLatencyMetrics,
    format_latency_summary,
)
from src.ferret_gaze.realtime.publisher import (
    NoOpRealtimePublisher,
    RealtimePublisher,
    create_realtime_publisher,
)
from src.ferret_gaze.realtime.scaffold_runner import run_realtime_transport_scaffold


class _CollectingPublisher(RealtimePublisher):
    """Collect packets in-memory for deterministic scaffold assertions."""

    def __init__(self) -> None:
        self.packets: list[RealtimeGazePacket] = []
        self.closed = False

    def publish(self, packet: RealtimeGazePacket) -> None:
        self.packets.append(packet)

    def close(self) -> None:
        self.closed = True


def _sample_packet() -> RealtimeGazePacket:
    """Build a minimal valid packet for schema tests."""
    return RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=1,
        publish_utc_ns=2,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(1.0, 0.0, 0.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(1.0, 0.0, 0.0),
        confidence=1.0,
    )


def test_packet_schema_accepts_valid_payload() -> None:
    packet = _sample_packet()
    assert packet.seq == 0
    assert packet.capture_utc_ns == 1
    assert packet.process_start_ns == 1
    assert packet.confidence == 1.0


def test_packet_schema_rejects_bad_quaternion_length() -> None:
    with pytest.raises(ValueError):
        RealtimeGazePacket(
            seq=0,
            capture_utc_ns=1,
            process_start_ns=1,
            publish_utc_ns=2,
            skull_position_xyz=(0.0, 0.0, 0.0),
            skull_quaternion_wxyz=(1.0, 0.0, 0.0),  # invalid length
            left_eye_origin_xyz=(0.0, 0.0, 0.0),
            left_gaze_direction_xyz=(1.0, 0.0, 0.0),
            right_eye_origin_xyz=(0.0, 0.0, 0.0),
            right_gaze_direction_xyz=(1.0, 0.0, 0.0),
            confidence=1.0,
        )


def test_publisher_factory_returns_noop_backend() -> None:
    publisher = create_realtime_publisher("noop")
    assert isinstance(publisher, NoOpRealtimePublisher)


def test_publisher_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_realtime_publisher("unknown")


def test_scaffold_runner_emits_requested_packet_count() -> None:
    publisher = _CollectingPublisher()
    summary = run_realtime_transport_scaffold(publisher=publisher, n_packets=5, hz=10000.0)
    assert len(publisher.packets) == 5
    assert publisher.closed is True
    assert [p.seq for p in publisher.packets] == [0, 1, 2, 3, 4]
    assert summary.packet_count == 5


def test_scaffold_runner_rejects_nonpositive_n_packets() -> None:
    with pytest.raises(ValueError):
        run_realtime_transport_scaffold(
            publisher=_CollectingPublisher(),
            n_packets=0,
            hz=60.0,
        )


def test_latency_metrics_summary_and_format() -> None:
    metrics = RealtimeLatencyMetrics(stale_threshold_ms=80.0)
    for seq in (0, 1, 3):
        packet = RealtimeGazePacket(
            seq=seq,
            capture_utc_ns=1_000_000_000,
            process_start_ns=1_020_000_000,
            publish_utc_ns=1_030_000_000,
            skull_position_xyz=(0.0, 0.0, 0.0),
            skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            left_eye_origin_xyz=(0.0, 0.0, 0.0),
            left_gaze_direction_xyz=(1.0, 0.0, 0.0),
            right_eye_origin_xyz=(0.0, 0.0, 0.0),
            right_gaze_direction_xyz=(1.0, 0.0, 0.0),
            confidence=1.0,
        )
        metrics.observe(packet=packet, now_utc_ns=1_090_000_000)
    summary = metrics.summary()
    assert summary.packet_count == 3
    assert summary.dropped_count == 1
    assert summary.stale_count == 3
    assert summary.end_to_end_p95_ms > 0.0
    assert summary.process_p95_ms > 0.0
    assert "p50/p95/p99" in format_latency_summary(summary)
