"""
Step 6 scaffold: per-frame realtime compute path.

This module defines pluggable interfaces for inference, triangulation, rolling
calibration, and gaze fusion. Current implementations are deterministic stubs
so integration can proceed before production runtimes are finalized.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket


@dataclass(frozen=True)
class FrameInferenceResult:
    """Stub 2D keypoint output for one frame."""

    seq: int
    confidence: float


@dataclass(frozen=True)
class TriangulationResult:
    """Stub 3D triangulation output for one frame."""

    seq: int
    skull_position_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class RollingCalibrationState:
    """Rolling eye-calibration state snapshot."""

    gain: float
    offset: float


class RealtimeInferenceRuntime(ABC):
    """Interface for per-frame realtime inference backends."""

    @abstractmethod
    def infer(self, packet: RealtimeGazePacket) -> FrameInferenceResult:
        """Run inference for one frame."""


class RealtimeTriangulator(ABC):
    """Interface for per-frame triangulation backends."""

    @abstractmethod
    def triangulate(self, inference: FrameInferenceResult) -> TriangulationResult:
        """Triangulate one frame from inference output."""


class RollingEyeCalibrator(ABC):
    """Interface for rolling eye-calibration backends."""

    @abstractmethod
    def update(self, triangulated: TriangulationResult) -> RollingCalibrationState:
        """Update calibrator state with one triangulation sample."""


class RealtimeGazeFuser(ABC):
    """Interface for per-frame gaze fusion backends."""

    @abstractmethod
    def fuse(
        self,
        packet: RealtimeGazePacket,
        triangulated: TriangulationResult,
        calibration: RollingCalibrationState,
        inference: FrameInferenceResult,
    ) -> RealtimeGazePacket:
        """Fuse per-frame compute outputs into a published gaze packet."""


class StubInferenceRuntime(RealtimeInferenceRuntime):
    """Deterministic confidence generator for scaffold bring-up."""

    def infer(self, packet: RealtimeGazePacket) -> FrameInferenceResult:
        confidence = max(0.0, min(1.0, 0.92 + 0.05 * math.sin(packet.seq * 0.05)))
        return FrameInferenceResult(seq=packet.seq, confidence=confidence)


class OnnxInferenceRuntime(RealtimeInferenceRuntime):
    """
    ONNX Runtime-backed inference adapter.

    This implementation keeps output shape minimal for current scaffolding by
    deriving one confidence value from model outputs. If model execution fails,
    it falls back to deterministic confidence so realtime loops stay alive.
    """

    def __init__(self, model_path: Path, provider: str = "CPUExecutionProvider") -> None:
        self._session = None
        self._input_name = None
        self._provider = provider
        self._model_path = model_path
        self._load_session()

    def _load_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("ONNX backend requires onnxruntime. Install with: uv add onnxruntime") from exc

        if not self._model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=[self._provider],
        )
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError(f"ONNX model has no inputs: {self._model_path}")
        self._input_name = inputs[0].name

    def infer(self, packet: RealtimeGazePacket) -> FrameInferenceResult:
        # Keep model input simple: seq and capture timestamp as float features.
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("ONNX backend requires numpy. Install with: uv add numpy") from exc

        try:
            features = np.array([[float(packet.seq), float(packet.capture_utc_ns)]], dtype=np.float32)
            outputs = self._session.run(None, {self._input_name: features})
            score = float(np.ravel(outputs[0])[0]) if outputs else 0.5
            confidence = max(0.0, min(1.0, score))
        except Exception:
            # Preserve realtime continuity if model/input mismatch occurs.
            confidence = max(0.0, min(1.0, 0.9 + 0.03 * math.sin(packet.seq * 0.05)))
        return FrameInferenceResult(seq=packet.seq, confidence=confidence)


class TensorRtInferenceRuntime(RealtimeInferenceRuntime):
    """
    TensorRT inference adapter (pluggable placeholder).

    Kept explicit so backend switching does not change call sites. This raises
    until a concrete engine-loading path is wired.
    """

    def __init__(self, engine_path: Path) -> None:
        self._engine_path = engine_path

    def infer(self, packet: RealtimeGazePacket) -> FrameInferenceResult:
        raise NotImplementedError(
            "TensorRT backend is pluggable but not yet implemented; "
            "use ONNX backend for now."
        )


class StubTriangulator(RealtimeTriangulator):
    """Deterministic 3D reconstruction stub."""

    def triangulate(self, inference: FrameInferenceResult) -> TriangulationResult:
        x = 0.1 * math.sin(inference.seq * 0.03)
        y = 0.1 * math.cos(inference.seq * 0.03)
        z = 0.02 * math.sin(inference.seq * 0.02)
        return TriangulationResult(seq=inference.seq, skull_position_xyz=(x, y, z))


class StubRollingEyeCalibrator(RollingEyeCalibrator):
    """Simple EWMA-style calibration state update."""

    def __init__(self) -> None:
        self._gain = 1.0
        self._offset = 0.0

    def update(self, triangulated: TriangulationResult) -> RollingCalibrationState:
        self._gain = (0.995 * self._gain) + (0.005 * (1.0 + abs(triangulated.skull_position_xyz[0])))
        self._offset = (0.99 * self._offset) + (0.01 * triangulated.skull_position_xyz[2])
        return RollingCalibrationState(gain=self._gain, offset=self._offset)


class StubGazeFuser(RealtimeGazeFuser):
    """Apply triangulation + calibration stubs into outgoing gaze packet."""

    def fuse(
        self,
        packet: RealtimeGazePacket,
        triangulated: TriangulationResult,
        calibration: RollingCalibrationState,
        inference: FrameInferenceResult,
    ) -> RealtimeGazePacket:
        x, y, z = triangulated.skull_position_xyz
        lx, ly, lz = packet.left_gaze_direction_xyz
        rx, ry, rz = packet.right_gaze_direction_xyz
        gain = calibration.gain
        offset = calibration.offset
        return packet.model_copy(
            update={
                "skull_position_xyz": (x, y, z),
                "left_gaze_direction_xyz": (lx * gain, ly, lz + offset),
                "right_gaze_direction_xyz": (rx * gain, ry, rz + offset),
                "confidence": inference.confidence,
            }
        )


def run_realtime_compute_scaffold(
    packets: list[RealtimeGazePacket],
    inference_runtime: RealtimeInferenceRuntime | None = None,
    triangulator: RealtimeTriangulator | None = None,
    calibrator: RollingEyeCalibrator | None = None,
    fuser: RealtimeGazeFuser | None = None,
) -> list[RealtimeGazePacket]:
    """Run per-frame compute scaffold over packets and return fused outputs."""
    if not packets:
        return []
    inference_runtime = inference_runtime or StubInferenceRuntime()
    triangulator = triangulator or StubTriangulator()
    calibrator = calibrator or StubRollingEyeCalibrator()
    fuser = fuser or StubGazeFuser()

    fused: list[RealtimeGazePacket] = []
    for packet in packets:
        inference = inference_runtime.infer(packet)
        triangulated = triangulator.triangulate(inference)
        calibration = calibrator.update(triangulated)
        fused_packet = fuser.fuse(packet, triangulated, calibration, inference)
        fused.append(fused_packet)
    return fused


def create_inference_runtime(
    backend: Literal["stub", "onnx", "tensorrt"] = "stub",
    model_path: Path | None = None,
    provider: str = "CPUExecutionProvider",
) -> RealtimeInferenceRuntime:
    """
    Build inference runtime from backend selection.
    """
    if backend == "stub":
        return StubInferenceRuntime()
    if model_path is None:
        raise ValueError("model_path is required for non-stub inference backends")
    if backend == "onnx":
        return OnnxInferenceRuntime(model_path=model_path, provider=provider)
    if backend == "tensorrt":
        return TensorRtInferenceRuntime(engine_path=model_path)
    raise ValueError(f"Unsupported inference backend: {backend}")
