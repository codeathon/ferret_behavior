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
