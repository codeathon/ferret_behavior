"""
Tests for step-6 per-frame realtime compute scaffold.
"""

import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

from src.ferret_gaze.realtime.per_frame_compute import (
    KeypointCentroidTriangulator,
    TensorRtInferenceRuntime,
    OnnxInferenceRuntime,
    FrameInferenceResult,
    create_inference_runtime,
    run_realtime_compute_scaffold,
)
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


def test_inference_factory_requires_model_path_for_non_stub() -> None:
    with pytest.raises(ValueError):
        create_inference_runtime(backend="onnx", model_path=None)


def test_tensorrt_runtime_is_pluggable_not_implemented() -> None:
    runtime = create_inference_runtime(backend="tensorrt", model_path=Path("engine.plan"))
    assert isinstance(runtime, TensorRtInferenceRuntime)
    packet = build_synthetic_replay_packets(n_packets=1)[0]
    with pytest.raises(NotImplementedError):
        runtime.infer(packet)


def test_keypoint_centroid_triangulator_uses_keypoint_geometry() -> None:
    triangulator = KeypointCentroidTriangulator()
    inference = FrameInferenceResult(
        seq=42,
        confidence=0.95,
        keypoints_xyz=((1.0, 2.0, 3.0), (4.0, 5.0, 9.0)),
    )
    triangulated = triangulator.triangulate(inference)
    assert triangulated.seq == 42
    assert triangulated.skull_position_xyz == (2.5, 3.5, 6.0)


def test_onnx_runtime_parses_confidence_and_xyz_keypoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "fake_model.onnx"
    model_path.write_text("fake-model")

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            self.providers = providers

        def get_inputs(self):
            return [SimpleNamespace(name="input_0")]

        def get_outputs(self):
            return [SimpleNamespace(name="output_0")]

        def run(self, output_names, feed_dict):
            assert output_names == ["output_0"]
            assert "input_0" in feed_dict
            # [confidence, x1, y1, z1, x2, y2, z2]
            return [[0.77, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = OnnxInferenceRuntime(model_path=model_path, provider="CPUExecutionProvider")
    packet = build_synthetic_replay_packets(n_packets=1)[0]
    result = runtime.infer(packet)

    assert result.seq == packet.seq
    assert result.confidence == pytest.approx(0.77)
    assert result.keypoints_xyz == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
