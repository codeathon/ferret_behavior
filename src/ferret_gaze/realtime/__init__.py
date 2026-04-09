"""Realtime gaze transport scaffold package."""

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.latency_metrics import (
    LatencySummary,
    RealtimeLatencyMetrics,
    format_latency_summary,
)
from src.ferret_gaze.realtime.publisher import (
    NoOpRealtimePublisher,
    RealtimePublisher,
    ZmqRealtimePublisher,
    create_realtime_publisher,
)
from src.ferret_gaze.realtime.scaffold_runner import (
    build_synthetic_replay_packets,
    run_realtime_transport_scaffold,
)
from src.ferret_gaze.realtime.solver_benchmark import (
    RealtimeSkullSolver,
    SlidingWindowCeresSkullSolverStub,
    SolverBenchmarkComparison,
    SolverBenchmarkStats,
    UkfRealtimeSkullSolverStub,
    benchmark_solver_against_reference,
    compare_stub_solvers,
)

__all__ = [
    "RealtimeGazePacket",
    "LatencySummary",
    "RealtimeLatencyMetrics",
    "RealtimePublisher",
    "NoOpRealtimePublisher",
    "ZmqRealtimePublisher",
    "create_realtime_publisher",
    "format_latency_summary",
    "build_synthetic_replay_packets",
    "run_realtime_transport_scaffold",
    "RealtimeSkullSolver",
    "SolverBenchmarkStats",
    "SolverBenchmarkComparison",
    "UkfRealtimeSkullSolverStub",
    "SlidingWindowCeresSkullSolverStub",
    "benchmark_solver_against_reference",
    "compare_stub_solvers",
]
