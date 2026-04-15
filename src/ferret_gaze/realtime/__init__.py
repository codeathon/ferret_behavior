"""Realtime gaze transport scaffold package."""

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.packet_serialize import gaze_packet_to_wire_dict
from src.ferret_gaze.realtime.calibration_projection import (
    SessionMultiViewCalibration,
    discover_session_calibration_toml,
    load_session_multi_view_calibration,
    projection_matrix_from_cam_block,
)
from src.ferret_gaze.realtime.latency_metrics import (
    LatencySummary,
    RealtimeLatencyMetrics,
    format_latency_summary,
)
from src.ferret_gaze.realtime.multiview_triangulation import triangulate_linear_dlt
from src.ferret_gaze.realtime.per_frame_compute import (
    OnnxImagesInferenceRuntime,
    OnnxInferenceRuntime,
    FrameInferenceResult,
    KeypointCentroidTriangulator,
    MultiviewOpenCvTriangulator,
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
from src.ferret_gaze.realtime.live_mocap_pipeline import (
    basler_frameset_to_live_mocap_frame_set,
    build_synthetic_live_mocap_frame_sets,
    gaze_packet_from_live_mocap_frame_set,
    run_live_mocap_compute_publish_session,
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
    "SessionMultiViewCalibration",
    "discover_session_calibration_toml",
    "load_session_multi_view_calibration",
    "projection_matrix_from_cam_block",
    "LatencySummary",
    "RealtimeLatencyMetrics",
    "FrameInferenceResult",
    "KeypointCentroidTriangulator",
    "MultiviewOpenCvTriangulator",
    "triangulate_linear_dlt",
    "TriangulationResult",
    "RollingCalibrationState",
    "RealtimeInferenceRuntime",
    "RealtimeTriangulator",
    "RollingEyeCalibrator",
    "RealtimeGazeFuser",
    "OnnxImagesInferenceRuntime",
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
    "gaze_packet_from_live_mocap_frame_set",
    "basler_frameset_to_live_mocap_frame_set",
    "build_synthetic_live_mocap_frame_sets",
    "run_live_mocap_compute_publish_session",
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
