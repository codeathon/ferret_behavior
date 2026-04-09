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
from src.ferret_gaze.realtime.publisher import RealtimePublisher
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _build_synthetic_packet(seq: int) -> RealtimeGazePacket:
    """Build one deterministic synthetic packet for transport bring-up."""
    capture_utc_ns = time.time_ns()
    yaw = 0.05 * math.sin(seq * 0.1)

    return RealtimeGazePacket(
        seq=seq,
        capture_utc_ns=capture_utc_ns,
        publish_utc_ns=time.time_ns(),
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, yaw, 0.0),
        left_eye_origin_xyz=(-10.0, 25.0, 0.0),
        left_gaze_direction_xyz=(1.0, 0.0, 0.0),
        right_eye_origin_xyz=(10.0, 25.0, 0.0),
        right_gaze_direction_xyz=(1.0, 0.0, 0.0),
        confidence=1.0,
    )


def run_realtime_transport_scaffold(
    publisher: RealtimePublisher,
    n_packets: int = 120,
    hz: float = 60.0,
) -> None:
    """
    Publish synthetic packets at a fixed rate using the configured publisher.

    This function is intentionally independent from live acquisition and solver
    code so we can validate Unreal transport integration first.
    """
    if n_packets <= 0:
        logger.warning("Realtime scaffold requested with n_packets=%d; nothing to send", n_packets)
        return
    if hz <= 0:
        raise ValueError("hz must be positive")

    period_s = 1.0 / hz
    logger.info(
        "Starting realtime transport scaffold: packets=%d, hz=%.2f",
        n_packets,
        hz,
    )

    try:
        for seq in range(n_packets):
            start = time.perf_counter()
            packet = _build_synthetic_packet(seq=seq)
            publisher.publish(packet)

            elapsed = time.perf_counter() - start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        publisher.close()
        logger.info("Realtime transport scaffold finished")
