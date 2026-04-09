"""
Synthetic realtime transport scaffold runner.

Step 2 goal: validate transport boundaries independent of camera/solver stages.
This runner emits deterministic synthetic packets through the configured
publisher backend.
"""

from __future__ import annotations

import math
import time

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.latency_metrics import (
    LatencySummary,
    RealtimeLatencyMetrics,
    format_latency_summary,
)
from src.ferret_gaze.realtime.publisher import RealtimePublisher
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _build_synthetic_packet(seq: int) -> RealtimeGazePacket:
    """Build one deterministic synthetic packet for transport bring-up."""
    capture_utc_ns = time.time_ns()
    process_start_ns = time.time_ns()
    yaw = 0.05 * math.sin(seq * 0.1)

    return RealtimeGazePacket(
        seq=seq,
        capture_utc_ns=capture_utc_ns,
        process_start_ns=process_start_ns,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, yaw, 0.0),
        left_eye_origin_xyz=(-10.0, 25.0, 0.0),
        left_gaze_direction_xyz=(1.0, 0.0, 0.0),
        right_eye_origin_xyz=(10.0, 25.0, 0.0),
        right_gaze_direction_xyz=(1.0, 0.0, 0.0),
        confidence=1.0,
    )


def build_synthetic_replay_packets(n_packets: int) -> list[RealtimeGazePacket]:
    """Create deterministic synthetic packet stream for replay benchmarks."""
    if n_packets <= 0:
        raise ValueError("n_packets must be positive")
    return [_build_synthetic_packet(seq=seq) for seq in range(n_packets)]


def run_realtime_transport_scaffold(
    publisher: RealtimePublisher,
    n_packets: int = 120,
    hz: float = 60.0,
    stale_threshold_ms: float = 80.0,
) -> LatencySummary:
    """
    Publish synthetic packets at a fixed rate using the configured publisher.

    This function is intentionally independent from live acquisition and solver
    code so we can validate Unreal transport integration first.
    """
    if n_packets <= 0:
        raise ValueError("n_packets must be positive")
    if hz <= 0:
        raise ValueError("hz must be positive")

    metrics = RealtimeLatencyMetrics(stale_threshold_ms=stale_threshold_ms)
    period_s = 1.0 / hz
    logger.info(
        "Starting realtime transport scaffold: packets=%d, hz=%.2f, stale_threshold_ms=%.1f",
        n_packets,
        hz,
        stale_threshold_ms,
    )

    try:
        for seq in range(n_packets):
            start = time.perf_counter()
            packet = _build_synthetic_packet(seq=seq)
            packet.publish_utc_ns = time.time_ns()
            publisher.publish(packet)
            metrics.observe(packet=packet, now_utc_ns=time.time_ns())

            elapsed = time.perf_counter() - start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        publisher.close()
    summary = metrics.summary()
    logger.info(format_latency_summary(summary))
    logger.info("Realtime transport scaffold finished")
    return summary
