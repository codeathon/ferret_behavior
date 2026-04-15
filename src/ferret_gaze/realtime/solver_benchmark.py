"""
Step 4 scaffold: realtime solver benchmark gate with stub adapters.

This module defines a pluggable realtime skull solver interface and a replay
benchmark runner so UKF-vs-Ceres decisions can be deferred until real solver
implementations are ready.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket


@dataclass(frozen=True)
class SolverBenchmarkStats:
    """Summary stats for one solver over a replay stream."""

    solver_name: str
    packet_count: int
    mean_solver_latency_ms: float
    p95_solver_latency_ms: float
    mean_position_error_mm: float
    mean_quaternion_l1_error: float


@dataclass(frozen=True)
class SolverBenchmarkComparison:
    """Pairwise benchmark output for UKF-vs-Ceres decision review."""

    ukf: SolverBenchmarkStats
    ceres: SolverBenchmarkStats
    recommended_solver: str
    recommendation_reason: str


class RealtimeSkullSolver(ABC):
    """Interface for realtime skull solver adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""

    @abstractmethod
    def solve(self, packet: RealtimeGazePacket) -> RealtimeGazePacket:
        """Return a solved packet for the given frame."""

    def solve_with_context(
        self,
        packet: RealtimeGazePacket,
        *,
        inference: object | None = None,
        triangulated: object | None = None,
    ) -> RealtimeGazePacket:
        """
        Optional live path: refine pose using inference / triangulation context.

        Default implementation ignores context and calls :meth:`solve` (replay benchmark).
        """
        _ = (inference, triangulated)
        return self.solve(packet)


class UkfRealtimeSkullSolverStub(RealtimeSkullSolver):
    """
    UKF stub adapter.

    Adds tiny deterministic skull-pose perturbations and a short compute delay
    to emulate a low-latency filtered solver.
    """

    def __init__(self, latency_ms: float = 1.0) -> None:
        self._latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "ukf_stub"

    def solve(self, packet: RealtimeGazePacket) -> RealtimeGazePacket:
        # Simulate solver compute cost.
        time.sleep(max(0.0, self._latency_ms) / 1000.0)
        x, y, z = packet.skull_position_xyz
        w, qx, qy, qz = packet.skull_quaternion_wxyz
        return packet.model_copy(
            update={
                "skull_position_xyz": (x + 0.05, y - 0.03, z + 0.01),
                "skull_quaternion_wxyz": (w, qx + 0.0005, qy - 0.0003, qz + 0.0002),
            }
        )


class SlidingWindowCeresSkullSolverStub(RealtimeSkullSolver):
    """
    Sliding-window Ceres stub adapter.

    Adds slightly lower geometric perturbation but higher compute delay to
    emulate optimization-window behavior.
    """

    def __init__(self, latency_ms: float = 4.0) -> None:
        self._latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "sliding_window_ceres_stub"

    def solve(self, packet: RealtimeGazePacket) -> RealtimeGazePacket:
        # Simulate optimization solve time.
        time.sleep(max(0.0, self._latency_ms) / 1000.0)
        x, y, z = packet.skull_position_xyz
        w, qx, qy, qz = packet.skull_quaternion_wxyz
        return packet.model_copy(
            update={
                "skull_position_xyz": (x + 0.02, y - 0.01, z + 0.005),
                "skull_quaternion_wxyz": (w, qx + 0.0002, qy - 0.0001, qz + 0.0001),
            }
        )


def _percentile(values: list[float], percentile: float) -> float:
    """Return nearest-rank percentile for non-empty float list."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = int(percentile * len(sorted_values))
    if rank >= len(sorted_values):
        rank = len(sorted_values) - 1
    return sorted_values[rank]


def _mean(values: list[float]) -> float:
    """Return arithmetic mean or 0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _position_error_mm(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Simple L1 position error in millimeters."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _quat_l1_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Simple quaternion L1 error in project wxyz convention."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])


def benchmark_solver_against_reference(
    solver: RealtimeSkullSolver,
    replay_packets: Iterable[RealtimeGazePacket],
    reference_packets: Iterable[RealtimeGazePacket],
) -> SolverBenchmarkStats:
    """Replay packet stream through one solver and summarize lag/geometry deltas."""
    replay_list = list(replay_packets)
    reference_list = list(reference_packets)
    if len(replay_list) != len(reference_list):
        raise ValueError("replay_packets and reference_packets must have equal length")
    if not replay_list:
        raise ValueError("benchmark requires at least one replay packet")

    latency_ms: list[float] = []
    pos_errors: list[float] = []
    quat_errors: list[float] = []

    for packet, reference in zip(replay_list, reference_list):
        start = time.perf_counter_ns()
        solved = solver.solve(packet)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        latency_ms.append(elapsed_ms)
        pos_errors.append(_position_error_mm(solved.skull_position_xyz, reference.skull_position_xyz))
        quat_errors.append(_quat_l1_error(solved.skull_quaternion_wxyz, reference.skull_quaternion_wxyz))

    return SolverBenchmarkStats(
        solver_name=solver.name,
        packet_count=len(replay_list),
        mean_solver_latency_ms=_mean(latency_ms),
        p95_solver_latency_ms=_percentile(latency_ms, 0.95),
        mean_position_error_mm=_mean(pos_errors),
        mean_quaternion_l1_error=_mean(quat_errors),
    )


def compare_stub_solvers(
    replay_packets: Iterable[RealtimeGazePacket],
    reference_packets: Iterable[RealtimeGazePacket],
) -> SolverBenchmarkComparison:
    """
    Run both stub solvers on identical replay data and choose a provisional
    recommendation via weighted latency/geometry score.
    """
    ukf_stats = benchmark_solver_against_reference(
        solver=UkfRealtimeSkullSolverStub(),
        replay_packets=replay_packets,
        reference_packets=reference_packets,
    )
    ceres_stats = benchmark_solver_against_reference(
        solver=SlidingWindowCeresSkullSolverStub(),
        replay_packets=replay_packets,
        reference_packets=reference_packets,
    )

    # Weight geometric consistency higher than latency until real data arrives.
    ukf_score = (2.0 * ukf_stats.mean_position_error_mm) + ukf_stats.p95_solver_latency_ms
    ceres_score = (2.0 * ceres_stats.mean_position_error_mm) + ceres_stats.p95_solver_latency_ms

    if ukf_score <= ceres_score:
        recommendation = ukf_stats.solver_name
    else:
        recommendation = ceres_stats.solver_name

    reason = (
        "Lower weighted score where geometric consistency is prioritized over latency "
        "(weights: geometry x2, latency x1)."
    )
    return SolverBenchmarkComparison(
        ukf=ukf_stats,
        ceres=ceres_stats,
        recommended_solver=recommendation,
        recommendation_reason=reason,
    )
