"""
Step 6 scaffold: per-frame realtime compute path.

This module defines pluggable interfaces for inference, triangulation, rolling
calibration, and gaze fusion. Current implementations are deterministic stubs
so integration can proceed before production runtimes are finalized.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.ferret_gaze.realtime.calibration_projection import (
    SessionMultiViewCalibration,
    load_session_multi_view_calibration,
)
from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.multiview_triangulation import triangulate_linear_dlt


@dataclass(frozen=True)
class FrameInferenceResult:
    """Per-frame inference output with confidence and optional keypoints."""

    seq: int
    confidence: float
    keypoints_xyz: tuple[tuple[float, float, float], ...] = ()
    # Optional multi-view 2D pixels for one skull proxy landmark: (cam_index, u, v) per view.
    single_landmark_uv_by_cam: tuple[tuple[int, float, float], ...] = ()


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
    def infer(
        self,
        packet: RealtimeGazePacket,
        *,
        frame_set: LiveMocapFrameSet | None = None,
    ) -> FrameInferenceResult:
        """
        Run inference for one frame.

        Image backends (e.g. ``onnx_images``) read ``frame_set.images_bgr``;
        others ignore ``frame_set``.
        """


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

    def infer(
        self,
        packet: RealtimeGazePacket,
        *,
        frame_set: LiveMocapFrameSet | None = None,
    ) -> FrameInferenceResult:
        confidence = max(0.0, min(1.0, 0.92 + 0.05 * math.sin(packet.seq * 0.05)))
        return FrameInferenceResult(seq=packet.seq, confidence=confidence, single_landmark_uv_by_cam=())


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
        self._output_names: list[str] = []
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
        self._output_names = [output.name for output in self._session.get_outputs()]

    def infer(
        self,
        packet: RealtimeGazePacket,
        *,
        frame_set: LiveMocapFrameSet | None = None,
    ) -> FrameInferenceResult:
        # Build a deterministic, explicit input tensor contract for realtime use.
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("ONNX backend requires numpy. Install with: uv add numpy") from exc

        try:
            # Features: [seq, capture_s, skull_xyz(3), left_gaze_xyz(3), right_gaze_xyz(3)].
            features = np.array(
                [[
                    float(packet.seq),
                    float(packet.capture_utc_ns) * 1e-9,
                    float(packet.skull_position_xyz[0]),
                    float(packet.skull_position_xyz[1]),
                    float(packet.skull_position_xyz[2]),
                    float(packet.left_gaze_direction_xyz[0]),
                    float(packet.left_gaze_direction_xyz[1]),
                    float(packet.left_gaze_direction_xyz[2]),
                    float(packet.right_gaze_direction_xyz[0]),
                    float(packet.right_gaze_direction_xyz[1]),
                    float(packet.right_gaze_direction_xyz[2]),
                ]],
                dtype=np.float32,
            )
            output_names = self._output_names or None
            outputs = self._session.run(output_names, {self._input_name: features})
            primary_output = np.ravel(outputs[0]) if outputs else np.array([0.5], dtype=np.float32)
            score = float(primary_output[0])
            confidence = max(0.0, min(1.0, score))

            # Optional geometry contract: first output may include xyz triplets after confidence.
            keypoints_xyz: tuple[tuple[float, float, float], ...] = ()
            if primary_output.size >= 4:
                remaining = primary_output[1:]
                valid_len = (remaining.size // 3) * 3
                reshaped = remaining[:valid_len].reshape(-1, 3)
                keypoints_xyz = tuple((float(x), float(y), float(z)) for x, y, z in reshaped)
        except Exception:
            # Preserve realtime continuity if model/input mismatch occurs.
            confidence = max(0.0, min(1.0, 0.9 + 0.03 * math.sin(packet.seq * 0.05)))
            keypoints_xyz = ()
        return FrameInferenceResult(
            seq=packet.seq,
            confidence=confidence,
            keypoints_xyz=keypoints_xyz,
            single_landmark_uv_by_cam=(),
        )


def _uv_model_to_original_pixels(
    u_m: float,
    v_m: float,
    orig_h: int,
    orig_w: int,
    model_h: int,
    model_w: int,
    output_uv_normalized: bool,
) -> tuple[float, float]:
    """Map model-space UVs to full-resolution image pixels for calibration alignment."""
    if output_uv_normalized:
        return float(u_m * orig_w), float(v_m * orig_h)
    su = orig_w / float(model_w)
    sv = orig_h / float(model_h)
    return float(u_m * su), float(v_m * sv)


class OnnxImagesInferenceRuntime(RealtimeInferenceRuntime):
    """
    Per-camera ONNX on BGR frames for one 2D landmark per view.

    Expects ``LiveMocapFrameSet`` on ``infer(..., frame_set=...)``. Runs the
    same model once per camera (sorted by index). Each forward uses input
    ``float32`` NCHW ``[1, 3, H, W]`` with values in ``[0, 1]`` (resize +
    RGB). Primary output is flattened: first two values are ``u``, then ``v``
    in either resized pixel space or normalized ``[0, 1]`` (see
    ``output_uv_normalized``); optional third value is a per-view confidence in
    ``[0, 1]``. Coordinates are scaled to the **original** image size before
    returning in ``single_landmark_uv_by_cam``.
    """

    def __init__(
        self,
        model_path: Path,
        provider: str = "CPUExecutionProvider",
        *,
        input_height: int = 256,
        input_width: int = 256,
        output_uv_normalized: bool = False,
    ) -> None:
        self._input_height = int(input_height)
        self._input_width = int(input_width)
        self._output_uv_normalized = bool(output_uv_normalized)
        self._model_path = model_path
        self._provider = provider
        self._session = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
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
        self._output_names = [output.name for output in self._session.get_outputs()]

    def _preprocess_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Resize BGR to ``HxW``, RGB, NCHW float32 in ``[0, 1]``."""
        resized = cv2.resize(
            image_bgr,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _fallback_confidence(self, seq: int) -> float:
        return max(0.0, min(1.0, 0.9 + 0.03 * math.sin(seq * 0.05)))

    def infer(
        self,
        packet: RealtimeGazePacket,
        *,
        frame_set: LiveMocapFrameSet | None = None,
    ) -> FrameInferenceResult:
        if frame_set is None or not frame_set.images_bgr:
            return FrameInferenceResult(
                seq=packet.seq,
                confidence=self._fallback_confidence(packet.seq),
                single_landmark_uv_by_cam=(),
            )

        per_view_conf: list[float] = []
        uv_rows: list[tuple[int, float, float]] = []

        for cam_id in sorted(frame_set.images_bgr.keys()):
            image_bgr = frame_set.images_bgr[cam_id]
            if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
                continue
            try:
                tensor = self._preprocess_bgr(image_bgr)
                assert self._session is not None and self._input_name is not None
                outs = self._session.run(
                    self._output_names or None,
                    {self._input_name: tensor},
                )
                flat = np.ravel(np.asarray(outs[0], dtype=np.float32))
                if flat.size < 2:
                    continue
                u_m, v_m = float(flat[0]), float(flat[1])
                oh, ow = int(image_bgr.shape[0]), int(image_bgr.shape[1])
                u_o, v_o = _uv_model_to_original_pixels(
                    u_m,
                    v_m,
                    oh,
                    ow,
                    self._input_height,
                    self._input_width,
                    self._output_uv_normalized,
                )
                uv_rows.append((cam_id, u_o, v_o))
                if flat.size >= 3:
                    per_view_conf.append(max(0.0, min(1.0, float(flat[2]))))
            except Exception:
                continue

        if not uv_rows:
            return FrameInferenceResult(
                seq=packet.seq,
                confidence=self._fallback_confidence(packet.seq),
                single_landmark_uv_by_cam=(),
            )

        confidence = (
            float(sum(per_view_conf) / len(per_view_conf))
            if per_view_conf
            else self._fallback_confidence(packet.seq)
        )
        return FrameInferenceResult(
            seq=packet.seq,
            confidence=confidence,
            single_landmark_uv_by_cam=tuple(uv_rows),
        )


class TensorRtInferenceRuntime(RealtimeInferenceRuntime):
    """
    TensorRT inference adapter (pluggable placeholder).

    Kept explicit so backend switching does not change call sites. This raises
    until a concrete engine-loading path is wired.
    """

    def __init__(self, engine_path: Path) -> None:
        self._engine_path = engine_path

    def infer(
        self,
        packet: RealtimeGazePacket,
        *,
        frame_set: LiveMocapFrameSet | None = None,
    ) -> FrameInferenceResult:
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


class MultiviewOpenCvTriangulator(RealtimeTriangulator):
    """
    Triangulate one landmark from multi-view pixels and session ``P`` matrices.

    Expects ``FrameInferenceResult.single_landmark_uv_by_cam`` with at least
    two views whose camera indices exist in the loaded calibration. Falls back
    to ``KeypointCentroidTriangulator`` when UV tracks are missing or too few
    views are available.
    """

    def __init__(self, calibration: SessionMultiViewCalibration) -> None:
        self._calibration = calibration
        self._fallback = KeypointCentroidTriangulator()

    def triangulate(self, inference: FrameInferenceResult) -> TriangulationResult:
        obs = inference.single_landmark_uv_by_cam
        if len(obs) < 2:
            return self._fallback.triangulate(inference)

        projections: list[np.ndarray] = []
        uv_pixels: list[np.ndarray] = []
        for cam_id, u, v in obs:
            if cam_id not in self._calibration.projection_by_cam_index:
                continue
            projections.append(self._calibration.projection_matrix(cam_id))
            uv_pixels.append(np.array([u, v], dtype=np.float64))
        if len(projections) < 2:
            return self._fallback.triangulate(inference)

        try:
            xyz = triangulate_linear_dlt(projections, uv_pixels)
        except (ValueError, np.linalg.LinAlgError):
            return self._fallback.triangulate(inference)

        return TriangulationResult(
            seq=inference.seq,
            skull_position_xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )


class KeypointCentroidTriangulator(RealtimeTriangulator):
    """Compute skull position from inference keypoint centroid when available."""

    def triangulate(self, inference: FrameInferenceResult) -> TriangulationResult:
        if not inference.keypoints_xyz:
            # Fall back to deterministic behavior to keep realtime loops running.
            x = 0.1 * math.sin(inference.seq * 0.03)
            y = 0.1 * math.cos(inference.seq * 0.03)
            z = 0.02 * math.sin(inference.seq * 0.02)
            return TriangulationResult(seq=inference.seq, skull_position_xyz=(x, y, z))
        xs = [point[0] for point in inference.keypoints_xyz]
        ys = [point[1] for point in inference.keypoints_xyz]
        zs = [point[2] for point in inference.keypoints_xyz]
        n = float(len(inference.keypoints_xyz))
        return TriangulationResult(
            seq=inference.seq,
            skull_position_xyz=(sum(xs) / n, sum(ys) / n, sum(zs) / n),
        )


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
    frame_sets: Sequence[LiveMocapFrameSet | None] | None = None,
) -> list[RealtimeGazePacket]:
    """
    Run per-frame compute scaffold over packets and return fused outputs.

    When ``frame_sets`` is provided with the same length as ``packets``, each
    element is passed to ``infer(..., frame_set=...)`` for image backends.
    """
    if not packets:
        return []
    inference_runtime = inference_runtime or StubInferenceRuntime()
    triangulator = triangulator or KeypointCentroidTriangulator()
    calibrator = calibrator or StubRollingEyeCalibrator()
    fuser = fuser or StubGazeFuser()

    fused: list[RealtimeGazePacket] = []
    for i, packet in enumerate(packets):
        fs: LiveMocapFrameSet | None = None
        if frame_sets is not None and i < len(frame_sets):
            fs = frame_sets[i]
        inference = inference_runtime.infer(packet, frame_set=fs)
        triangulated = triangulator.triangulate(inference)
        calibration = calibrator.update(triangulated)
        fused_packet = fuser.fuse(packet, triangulated, calibration, inference)
        fused.append(fused_packet)
    return fused


def create_inference_runtime(
    backend: Literal["stub", "onnx", "onnx_images", "tensorrt"] = "stub",
    model_path: Path | None = None,
    provider: str = "CPUExecutionProvider",
    *,
    images_input_height: int = 256,
    images_input_width: int = 256,
    output_uv_normalized: bool = False,
) -> RealtimeInferenceRuntime:
    """
    Build inference runtime from backend selection.

    ``onnx_images`` runs one landmark head per camera from
    :class:`~src.ferret_gaze.realtime.live_frame_set.LiveMocapFrameSet` images;
    resize size and UV interpretation are controlled by the keyword arguments.
    """
    if backend == "stub":
        return StubInferenceRuntime()
    if model_path is None:
        raise ValueError("model_path is required for non-stub inference backends")
    if backend == "onnx":
        return OnnxInferenceRuntime(model_path=model_path, provider=provider)
    if backend == "onnx_images":
        return OnnxImagesInferenceRuntime(
            model_path=model_path,
            provider=provider,
            input_height=images_input_height,
            input_width=images_input_width,
            output_uv_normalized=output_uv_normalized,
        )
    if backend == "tensorrt":
        return TensorRtInferenceRuntime(engine_path=model_path)
    raise ValueError(f"Unsupported inference backend: {backend}")


def create_triangulator(
    backend: Literal["keypoint_centroid", "stub", "multiview_opencv"] = "keypoint_centroid",
    *,
    calibration_toml_path: Path | None = None,
) -> RealtimeTriangulator:
    """
    Build triangulator from backend selection.

    ``multiview_opencv`` requires ``calibration_toml_path`` to a freemocap-style
    ``*camera_calibration.toml`` (see ``load_session_multi_view_calibration``).
    """
    if backend == "keypoint_centroid":
        return KeypointCentroidTriangulator()
    if backend == "stub":
        return StubTriangulator()
    if backend == "multiview_opencv":
        if calibration_toml_path is None:
            raise ValueError("calibration_toml_path is required for multiview_opencv triangulation backend")
        calib = load_session_multi_view_calibration(calibration_toml_path)
        return MultiviewOpenCvTriangulator(calibration=calib)
    raise ValueError(f"Unsupported triangulation backend: {backend}")
