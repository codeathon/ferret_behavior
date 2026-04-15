"""Realtime gaze transport scaffold package."""

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.packet_serialize import gaze_packet_to_wire_dict
from src.ferret_gaze.realtime.latency_metrics import (
    LatencySummary,
    RealtimeLatencyMetrics,
    format_latency_summary,
)
from src.ferret_gaze.realtime.per_frame_compute import (
    OnnxInferenceRuntime,
    FrameInferenceResult,
    KeypointCentroidTriangulator,
    RealtimeGazeFuser,
    RealtimeInferenceRuntime,
    RealtimeTriangulator,
    RollingCalibrationState,
    RollingEyeCalibrator,
    StubGazeFuser,
    StubInferenceRuntime,
    StubRollingEyeCalibrator,
    StubTriangulator,
    TensorRtInferenceRuntime,
    TriangulationResult,
    create_triangulator,
    create_inference_runtime,
    run_realtime_compute_scaffold,
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
from src.ferret_gaze.realtime.runtime_config import (
    RealtimeRuntimeConfig,
    load_realtime_runtime_config,
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
    "LiveMocapFrameSet",
    "gaze_packet_to_wire_dict",
    "LatencySummary",
    "RealtimeLatencyMetrics",
    "FrameInferenceResult",
    "KeypointCentroidTriangulator",
    "TriangulationResult",
    "RollingCalibrationState",
    "RealtimeInferenceRuntime",
    "RealtimeTriangulator",
    "RollingEyeCalibrator",
    "RealtimeGazeFuser",
    "OnnxInferenceRuntime",
    "TensorRtInferenceRuntime",
    "StubInferenceRuntime",
    "StubTriangulator",
    "StubRollingEyeCalibrator",
    "StubGazeFuser",
    "create_inference_runtime",
    "create_triangulator",
    "run_realtime_compute_scaffold",
    "RealtimePublisher",
    "NoOpRealtimePublisher",
    "ZmqRealtimePublisher",
    "create_realtime_publisher",
    "format_latency_summary",
    "build_synthetic_replay_packets",
    "run_realtime_transport_scaffold",
    "RealtimeRuntimeConfig",
    "load_realtime_runtime_config",
    "RealtimeSkullSolver",
    "SolverBenchmarkStats",
    "SolverBenchmarkComparison",
    "UkfRealtimeSkullSolverStub",
    "SlidingWindowCeresSkullSolverStub",
    "benchmark_solver_against_reference",
    "compare_stub_solvers",
]
