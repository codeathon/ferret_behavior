"""
Tests for step-6 per-frame realtime compute scaffold.
"""

from src.ferret_gaze.realtime.per_frame_compute import run_realtime_compute_scaffold
from src.ferret_gaze.realtime.scaffold_runner import build_synthetic_replay_packets


def test_compute_scaffold_returns_one_output_per_input() -> None:
    packets = build_synthetic_replay_packets(n_packets=10)
    outputs = run_realtime_compute_scaffold(packets)
    assert len(outputs) == len(packets)
    assert [packet.seq for packet in outputs] == [packet.seq for packet in packets]


def test_compute_scaffold_populates_confidence_and_updates_pose() -> None:
    packets = build_synthetic_replay_packets(n_packets=4)
    outputs = run_realtime_compute_scaffold(packets)
    for original, fused in zip(packets, outputs):
        assert fused.confidence is not None
        assert fused.skull_position_xyz != original.skull_position_xyz


def test_compute_scaffold_handles_empty_input() -> None:
    assert run_realtime_compute_scaffold([]) == []
