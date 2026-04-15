"""
Tests for step-6 per-frame realtime compute scaffold.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    KeypointCentroidTriangulator,
    OnnxImagesInferenceRuntime,
    OnnxInferenceRuntime,
    StubTriangulator,
    TensorRtInferenceRuntime,
    create_inference_runtime,
    create_triangulator,
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


def test_inference_factory_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_inference_runtime(backend="unknown", model_path=tmp_path / "m.onnx")  # type: ignore[arg-type]


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


def test_triangulator_factory_selects_keypoint_centroid() -> None:
    triangulator = create_triangulator(backend="keypoint_centroid")
    assert isinstance(triangulator, KeypointCentroidTriangulator)


def test_triangulator_factory_selects_stub() -> None:
    triangulator = create_triangulator(backend="stub")
    assert isinstance(triangulator, StubTriangulator)


def test_triangulator_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_triangulator(backend="unknown")  # type: ignore[arg-type]


def test_inference_factory_onnx_images_returns_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose.onnx"
    model_path.write_text("fake")

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            self.providers = providers

        def get_inputs(self):
            return [SimpleNamespace(name="in0")]

        def get_outputs(self):
            return [SimpleNamespace(name="out0")]

        def run(self, output_names, feed_dict):
            return [np.zeros((1, 3), dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    runtime = create_inference_runtime(
        backend="onnx_images",
        model_path=model_path,
        images_input_height=128,
        images_input_width=128,
        output_uv_normalized=True,
    )
    assert isinstance(runtime, OnnxImagesInferenceRuntime)


def test_onnx_images_infer_without_frame_set_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose.onnx"
    model_path.write_text("fake")

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="in0")]

        def get_outputs(self):
            return [SimpleNamespace(name="out0")]

        def run(self, output_names, feed_dict):
            raise AssertionError("should not run without images")

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    runtime = OnnxImagesInferenceRuntime(model_path=model_path, input_height=64, input_width=64)
    packet = build_synthetic_replay_packets(n_packets=1)[0]
    out = runtime.infer(packet, frame_set=None)
    assert out.seq == packet.seq
    assert out.single_landmark_uv_by_cam == ()
    assert 0.0 <= out.confidence <= 1.0


def test_onnx_images_infer_multi_cam_uvs_scaled_to_original_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model emits u,v in resized pixel space; returned UVs match full-res image coordinates."""
    model_path = tmp_path / "pose.onnx"
    model_path.write_text("fake")

    calls: list[np.ndarray] = []

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="in0")]

        def get_outputs(self):
            return [SimpleNamespace(name="out0")]

        def run(self, output_names, feed_dict):
            tensor = feed_dict["in0"]
            calls.append(tensor)
            # First cam: (10, 20) in 256x256 space; second: (30, 40)
            if len(calls) == 1:
                return [np.array([[10.0, 20.0, 0.7]], dtype=np.float32)]
            return [np.array([[30.0, 40.0, 0.8]], dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = OnnxImagesInferenceRuntime(
        model_path=model_path,
        input_height=256,
        input_width=256,
        output_uv_normalized=False,
    )
    # BGR images: h=100, w=200 per camera.
    img0 = np.zeros((100, 200, 3), dtype=np.uint8)
    img1 = np.zeros((100, 200, 3), dtype=np.uint8)
    fs = LiveMocapFrameSet(seq=0, anchor_utc_ns=1_000, images_bgr={0: img0, 1: img1})
    packet = build_synthetic_replay_packets(n_packets=1)[0]
    out = runtime.infer(packet, frame_set=fs)

    assert len(out.single_landmark_uv_by_cam) == 2
    u0, v0 = out.single_landmark_uv_by_cam[0][1], out.single_landmark_uv_by_cam[0][2]
    assert u0 == pytest.approx(10.0 * (200.0 / 256.0))
    assert v0 == pytest.approx(20.0 * (100.0 / 256.0))
    u1, v1 = out.single_landmark_uv_by_cam[1][1], out.single_landmark_uv_by_cam[1][2]
    assert u1 == pytest.approx(30.0 * (200.0 / 256.0))
    assert v1 == pytest.approx(40.0 * (100.0 / 256.0))
    assert out.confidence == pytest.approx((0.7 + 0.8) / 2.0)
    assert calls[0].shape == (1, 3, 256, 256)


def test_run_realtime_compute_scaffold_passes_frame_sets_to_infer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "pose.onnx"
    model_path.write_text("fake")

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="in0")]

        def get_outputs(self):
            return [SimpleNamespace(name="out0")]

        def run(self, output_names, feed_dict):
            return [np.array([[1.0, 2.0, 0.5]], dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = OnnxImagesInferenceRuntime(model_path=model_path, input_height=32, input_width=32)
    packets = build_synthetic_replay_packets(n_packets=2)
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    fs0 = LiveMocapFrameSet(seq=0, anchor_utc_ns=1, images_bgr={0: img})
    fs1 = LiveMocapFrameSet(seq=1, anchor_utc_ns=2, images_bgr={0: img})
    outs = run_realtime_compute_scaffold(
        packets,
        inference_runtime=runtime,
        frame_sets=[fs0, fs1],
    )
    assert len(outs) == 2
    # Triangulation updates skull from UV path via multiview not used — centroid/stub path;
    # at least confidence should reflect inference when fuser passes through.
    assert outs[0].confidence is not None
