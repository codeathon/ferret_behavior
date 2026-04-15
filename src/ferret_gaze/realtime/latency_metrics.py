"""
Latency instrumentation helpers for realtime transport scaffolding.

This module provides a small in-memory collector so realtime bring-up can report
stable summary statistics without introducing heavyweight dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket


@dataclass(frozen=True)
class LatencySummary:
    """Snapshot of realtime latency metrics for one scaffold run."""

    packet_count: int
    dropped_count: int
    stale_count: int
    end_to_end_p50_ms: float
    end_to_end_p95_ms: float
    end_to_end_p99_ms: float
    process_p50_ms: float
    process_p95_ms: float
    process_p99_ms: float
    stale_threshold_ms: float
    queue_overflow_count: int = 0
    stage_error_count: int = 0
    publish_error_count: int = 0


def _percentile_ms(samples_ns: list[int], percentile: float) -> float:
    """Return percentile in milliseconds using nearest-rank over integer ns."""
    if not samples_ns:
        return 0.0
    sorted_samples = sorted(samples_ns)
    rank = max(1, math.ceil(percentile * len(sorted_samples)))
    index = min(rank - 1, len(sorted_samples) - 1)
    return sorted_samples[index] / 1_000_000.0


class RealtimeLatencyMetrics:
    """Collect latency and packet health metrics from a realtime stream."""

    def __init__(self, stale_threshold_ms: float = 80.0) -> None:
        if stale_threshold_ms <= 0:
            raise ValueError("stale_threshold_ms must be positive")
        self._stale_threshold_ns = int(stale_threshold_ms * 1_000_000.0)
        self._stale_threshold_ms = stale_threshold_ms
        self._end_to_end_latency_ns: list[int] = []
        self._process_latency_ns: list[int] = []
        self._packet_count = 0
        self._stale_count = 0
        self._dropped_count = 0
        self._queue_overflow_count = 0
        self._stage_error_count = 0
        self._publish_error_count = 0
        self._last_seq: int | None = None

    def observe(self, packet: RealtimeGazePacket, now_utc_ns: int) -> None:
        """Update metrics with one packet observation at current wall-clock ns."""
        self._packet_count += 1
        self._observe_seq(packet.seq)

        end_to_end_ns = max(0, now_utc_ns - packet.capture_utc_ns)
        self._end_to_end_latency_ns.append(end_to_end_ns)
        if end_to_end_ns > self._stale_threshold_ns:
            self._stale_count += 1

        if packet.publish_utc_ns is not None and packet.process_start_ns is not None:
            process_ns = max(0, packet.publish_utc_ns - packet.process_start_ns)
            self._process_latency_ns.append(process_ns)

    def _observe_seq(self, seq: int) -> None:
        """Count sequence gaps as dropped packets."""
        if self._last_seq is not None and seq > self._last_seq + 1:
            self._dropped_count += seq - self._last_seq - 1
        self._last_seq = seq

    def record_queue_overflow(self, count: int = 1) -> None:
        """Count queue-overflow events from producer/consumer backpressure."""
        if count > 0:
            self._queue_overflow_count += int(count)

    def record_stage_error(self, count: int = 1) -> None:
        """Count per-frame compute failures (inference/triangulation/fuse)."""
        if count > 0:
            self._stage_error_count += int(count)

    def record_publish_error(self, count: int = 1) -> None:
        """Count publish backend failures while sending packets."""
        if count > 0:
            self._publish_error_count += int(count)

    def summary(self) -> LatencySummary:
        """Build immutable metrics summary for logging and tests."""
        return LatencySummary(
            packet_count=self._packet_count,
            dropped_count=self._dropped_count,
            stale_count=self._stale_count,
            end_to_end_p50_ms=_percentile_ms(self._end_to_end_latency_ns, 0.50),
            end_to_end_p95_ms=_percentile_ms(self._end_to_end_latency_ns, 0.95),
            end_to_end_p99_ms=_percentile_ms(self._end_to_end_latency_ns, 0.99),
            process_p50_ms=_percentile_ms(self._process_latency_ns, 0.50),
            process_p95_ms=_percentile_ms(self._process_latency_ns, 0.95),
            process_p99_ms=_percentile_ms(self._process_latency_ns, 0.99),
            stale_threshold_ms=self._stale_threshold_ms,
            queue_overflow_count=self._queue_overflow_count,
            stage_error_count=self._stage_error_count,
            publish_error_count=self._publish_error_count,
        )


def format_latency_summary(summary: LatencySummary) -> str:
    """Return a compact one-line summary string for logs/CLI visibility."""
    return (
        "Realtime latency summary: packets={packets}, dropped={dropped}, stale={stale}"
        ", queue_overflow={queue_overflow}, stage_errors={stage_errors}, publish_errors={publish_errors}"
        " (threshold_ms={threshold:.1f}), "
        "e2e_ms[p50/p95/p99]={e2e_p50:.2f}/{e2e_p95:.2f}/{e2e_p99:.2f}, "
        "process_ms[p50/p95/p99]={proc_p50:.2f}/{proc_p95:.2f}/{proc_p99:.2f}"
    ).format(
        packets=summary.packet_count,
        dropped=summary.dropped_count,
        stale=summary.stale_count,
        queue_overflow=summary.queue_overflow_count,
        stage_errors=summary.stage_error_count,
        publish_errors=summary.publish_error_count,
        threshold=summary.stale_threshold_ms,
        e2e_p50=summary.end_to_end_p50_ms,
        e2e_p95=summary.end_to_end_p95_ms,
        e2e_p99=summary.end_to_end_p99_ms,
        proc_p50=summary.process_p50_ms,
        proc_p95=summary.process_p95_ms,
        proc_p99=summary.process_p99_ms,
    )
