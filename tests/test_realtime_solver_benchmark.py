"""
Tests for step-4 realtime solver benchmark scaffolding.
"""

from src.ferret_gaze.realtime.scaffold_runner import build_synthetic_replay_packets
from src.ferret_gaze.realtime.solver_benchmark import (
    SlidingWindowCeresSkullSolverStub,
    UkfRealtimeSkullSolverStub,
    benchmark_solver_against_reference,
    compare_stub_solvers,
)


def test_build_synthetic_replay_packets_returns_requested_count() -> None:
    packets = build_synthetic_replay_packets(n_packets=6)
    assert len(packets) == 6
    assert [packet.seq for packet in packets] == [0, 1, 2, 3, 4, 5]


def test_stub_solver_benchmark_returns_nonzero_stats() -> None:
    replay_packets = build_synthetic_replay_packets(n_packets=8)
    reference_packets = [packet.model_copy(deep=True) for packet in replay_packets]
    stats = benchmark_solver_against_reference(
        solver=UkfRealtimeSkullSolverStub(latency_ms=0.0),
        replay_packets=replay_packets,
        reference_packets=reference_packets,
    )
    assert stats.packet_count == 8
    assert stats.p95_solver_latency_ms >= 0.0
    assert stats.mean_position_error_mm >= 0.0
    assert stats.mean_quaternion_l1_error >= 0.0


def test_compare_stub_solvers_returns_expected_solver_names() -> None:
    replay_packets = build_synthetic_replay_packets(n_packets=8)
    reference_packets = [packet.model_copy(deep=True) for packet in replay_packets]
    comparison = compare_stub_solvers(
        replay_packets=replay_packets,
        reference_packets=reference_packets,
    )
    assert comparison.ukf.solver_name == UkfRealtimeSkullSolverStub().name
    assert comparison.ceres.solver_name == SlidingWindowCeresSkullSolverStub().name
    assert comparison.recommended_solver in {
        UkfRealtimeSkullSolverStub().name,
        SlidingWindowCeresSkullSolverStub().name,
    }
