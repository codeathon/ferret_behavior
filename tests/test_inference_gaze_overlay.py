"""Tests for inference gaze / eye-origin fields and packet overlay."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.live_mocap_pipeline import process_live_mocap_tick
from src.ferret_gaze.realtime.per_frame_compute import (
    FrameInferenceResult,
    OnnxImagesInferenceRuntime,
    OnnxInferenceRuntime,
    RealtimeInferenceRuntime,
    StubGazeFuser,
    StubRollingEyeCalibrator,
    StubTriangulator,
    apply_inference_to_gaze_packet,
)


def test_apply_inference_unitizes_gaze_and_copies_origins() -> None:
    p = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(0.0, 0.0, 1.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 0.0, 1.0),
        confidence=1.0,
    )
    inf = FrameInferenceResult(
        seq=0,
        confidence=1.0,
        left_gaze_direction_xyz=(3.0, 0.0, 4.0),
        right_gaze_direction_xyz=(0.0, 5.0, 0.0),
        left_eye_origin_xyz=(-1.0, 2.0, 3.0),
        right_eye_origin_xyz=(4.0, 5.0, 6.0),
    )
    out = apply_inference_to_gaze_packet(p, inf)
    assert out.left_gaze_direction_xyz == pytest.approx((0.6, 0.0, 0.8))
    assert out.right_gaze_direction_xyz == pytest.approx((0.0, 1.0, 0.0))
    assert out.left_eye_origin_xyz == (-1.0, 2.0, 3.0)
    assert out.right_eye_origin_xyz == (4.0, 5.0, 6.0)


def test_onnx_runtime_reads_second_output_as_gaze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "fake_model.onnx"
    model_path.write_text("fake-model")

    class _FakeSession:
        def __init__(self, _: str, providers: list[str]) -> None:
            self.providers = providers

        def get_inputs(self):
            return [SimpleNamespace(name="input_0")]

        def get_outputs(self):
            return [SimpleNamespace(name="o0"), SimpleNamespace(name="o1")]

        def run(self, output_names, feed_dict):
            assert output_names == ["o0", "o1"]
            return [
                [0.88, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = OnnxInferenceRuntime(model_path=model_path, provider="CPUExecutionProvider")
    p = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(1.0, 0.0, 0.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 1.0, 0.0),
        confidence=1.0,
    )
    r = runtime.infer(p)
    assert r.confidence == pytest.approx(0.88)
    assert r.keypoints_xyz == ((1.0, 0.0, 0.0),)
    assert r.left_gaze_direction_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert r.right_gaze_direction_xyz == pytest.approx((0.0, 1.0, 0.0))


def test_onnx_images_extended_flat_includes_gaze(
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
            return [np.array([[1.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = OnnxImagesInferenceRuntime(model_path=model_path, input_height=64, input_width=64)
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    fs = LiveMocapFrameSet(seq=0, anchor_utc_ns=1, images_bgr={0: img})
    p = RealtimeGazePacket(
        seq=0,
        capture_utc_ns=1,
        process_start_ns=None,
        publish_utc_ns=None,
        skull_position_xyz=(0.0, 0.0, 0.0),
        skull_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        left_eye_origin_xyz=(0.0, 0.0, 0.0),
        left_gaze_direction_xyz=(0.0, 0.0, 1.0),
        right_eye_origin_xyz=(0.0, 0.0, 0.0),
        right_gaze_direction_xyz=(0.0, 0.0, 1.0),
        confidence=1.0,
    )
    out = runtime.infer(p, frame_set=fs)
    assert out.left_gaze_direction_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert out.right_gaze_direction_xyz == pytest.approx((0.0, 1.0, 0.0))


class _GazeInfer(RealtimeInferenceRuntime):
    def infer(self, packet, *, frame_set=None):
        return FrameInferenceResult(
            seq=packet.seq,
            confidence=1.0,
            left_gaze_direction_xyz=(1.0, 0.0, 0.0),
            right_gaze_direction_xyz=(0.0, 1.0, 0.0),
        )


def test_process_live_mocap_applies_inference_gaze_before_fuse() -> None:
    fs = LiveMocapFrameSet(seq=0, anchor_utc_ns=1_000, images_bgr={0: np.zeros((2, 2, 3), dtype=np.uint8)})
    fused = process_live_mocap_tick(
        fs,
        inference_runtime=_GazeInfer(),
        triangulator=StubTriangulator(),
        calibrator=StubRollingEyeCalibrator(),
        fuser=StubGazeFuser(),
        skull_solver=None,
    )
    # StubGazeFuser scales left x by rolling gain (~1); directions should reflect inference, not default +Z.
    assert fused.left_gaze_direction_xyz[0] > 0.5
    assert fused.right_gaze_direction_xyz[1] > 0.5
