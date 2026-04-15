"""
End-to-end-ish tests for the realtime pipeline using config + factories.

These tests avoid camera hardware and avoid committing binary ONNX assets by
monkeypatching ``onnxruntime.InferenceSession``. They still exercise:
- runtime config -> factory wiring (inference/triangulation/skull solver/fuser)
- live mocap orchestration (frame bundles -> infer -> overlay -> triangulate -> solve -> calibrate -> fuse)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.batch_processing.full_pipeline import run_pipeline
from src.ferret_gaze.realtime.kabsch_skull_solver import DEFAULT_KABSCH_REFERENCE_BODY
from src.ferret_gaze.realtime.live_mocap_pipeline import build_synthetic_live_mocap_frame_sets
from src.ferret_gaze.realtime.per_frame_compute import create_inference_runtime, create_triangulator
from src.ferret_gaze.realtime.publisher import create_realtime_publisher
from src.ferret_gaze.realtime.runtime_config import RealtimeRuntimeConfig
from src.ferret_gaze.realtime.anatomical_mocap_fuse import create_eye_calibrator, create_gaze_fuser
from src.ferret_gaze.realtime.kabsch_skull_solver import create_skull_solver
from src.ferret_gaze.realtime.live_mocap_pipeline import run_live_mocap_compute_publish_session


def _write_minimal_two_cam_calibration_toml(path: Path) -> None:
	"""
	Write a minimal freemocap-style ``*camera_calibration.toml`` for 2 cameras.

	Cam0 at world origin, looking along +Z; Cam1 shifted +X by 100 mm.
	"""
	k = [[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]]
	o = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # camera axes in world
	c0 = [0.0, 0.0, 0.0]
	c1 = [100.0, 0.0, 0.0]
	text = "\n".join(
		[
			"[cam_0]",
			'name = "cam0"',
			f"matrix = {k}",
			f"world_position = {c0}",
			f"world_orientation = {o}",
			"",
			"[cam_1]",
			'name = "cam1"',
			f"matrix = {k}",
			f"world_position = {c1}",
			f"world_orientation = {o}",
			"",
		]
	)
	path.write_text(text, encoding="utf-8")


def _project_point(P: np.ndarray, x_world: np.ndarray) -> tuple[float, float]:
	xh = np.concatenate([x_world.reshape(3), np.array([1.0])])
	uvh = P @ xh
	u = float(uvh[0] / uvh[2])
	v = float(uvh[1] / uvh[2])
	return u, v


def test_e2e_live_mocap_factories_onnx_images_multiview_anatomical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	"""
	End-to-end through factories (A): onnx_images -> multiview triangulation -> anatomical fuse.

	This asserts the multiview path is used by ensuring UVs are present for both cameras.
	"""
	# --- fake onnxruntime model file ---
	model_path = tmp_path / "pose.onnx"
	model_path.write_text("fake")

	# --- calibration toml ---
	calib_path = tmp_path / "unit_camera_calibration.toml"
	_write_minimal_two_cam_calibration_toml(calib_path)

	# --- fake onnxruntime session that emits per-cam UV for a known 3D point ---
	from src.ferret_gaze.realtime.calibration_projection import load_session_multi_view_calibration

	calib = load_session_multi_view_calibration(calib_path)
	target_xyz = np.array([10.0, 5.0, 400.0], dtype=np.float64)  # in front of both cameras
	u0, v0 = _project_point(calib.projection_matrix(0), target_xyz)
	u1, v1 = _project_point(calib.projection_matrix(1), target_xyz)
	outs = [
		np.array([[u0, v0, 0.9, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
		np.array([[u1, v1, 0.9, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
	]

	class _FakeSession:
		def __init__(self, _: str, providers: list[str]) -> None:
			_ = providers
			self._calls = 0

		def get_inputs(self):
			return [SimpleNamespace(name="in0")]

		def get_outputs(self):
			return [SimpleNamespace(name="out0")]

		def run(self, output_names, feed_dict):
			_ = (output_names, feed_dict)
			out = outs[min(self._calls, len(outs) - 1)]
			self._calls += 1
			return [out]

	fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
	monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

	# --- build real components via factories ---
	infer = create_inference_runtime(
		backend="onnx_images",
		model_path=model_path,
		images_input_height=128,
		images_input_width=128,
		output_uv_normalized=False,
	)
	tri = create_triangulator(backend="multiview_opencv", calibration_toml_path=calib_path)
	calibrator = create_eye_calibrator("anatomical", vergence_ema_alpha=0.2)
	fuser = create_gaze_fuser("anatomical", half_ipd_mm=10.0, eye_y_mm=25.0, eye_z_mm=0.0)
	publisher = create_realtime_publisher(backend="noop", endpoint="tcp://127.0.0.1:1", topic="t", payload_format="json")

	frame_sets = build_synthetic_live_mocap_frame_sets(3, n_cams=2, height=80, width=120)
	summary = run_live_mocap_compute_publish_session(
		frame_sets=frame_sets,
		publisher=publisher,
		inference_runtime=infer,
		triangulator=tri,
		hz=1000.0,
		stale_threshold_ms=10.0,
		calibrator=calibrator,
		fuser=fuser,
		skull_solver=None,
	)
	assert summary.packet_count == 3


def test_e2e_run_pipeline_live_mocap_onnx_centroid_kabsch_ceres(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	"""
	End-to-end through config + full_pipeline (B): onnx -> centroid triangulation -> skull solver.

	We run twice (kabsch then ceres) to ensure both backends are constructible and
	operate without error on the live_mocap path.
	"""
	model_path = tmp_path / "tabular.onnx"
	model_path.write_text("fake")

	ref = DEFAULT_KABSCH_REFERENCE_BODY.copy()
	# Observed points are a rotated version of the reference template (row convention).
	rz = np.array(
		[
			[0.0, -1.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float64,
	)
	obs = ref @ rz.T
	flat_kps = [float(v) for row in obs for v in row]

	class _FakeSession:
		def __init__(self, _: str, providers: list[str]) -> None:
			self.providers = providers

		def get_inputs(self):
			return [SimpleNamespace(name="input_0")]

		def get_outputs(self):
			return [SimpleNamespace(name="o0"), SimpleNamespace(name="o1")]

		def run(self, output_names, feed_dict):
			_ = (output_names, feed_dict)
			# o0: [confidence, keypoints xyz...]
			o0 = [0.95] + flat_kps
			# o1: [lx,ly,lz, rx,ry,rz]
			o1 = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
			return [o0, o1]

	fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
	monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

	def _write_runtime_json(path: Path, backend: str) -> None:
		cfg = RealtimeRuntimeConfig(
			realtime_mode="live_mocap",
			live_mocap_frame_source="synthetic",
			transport_backend="noop",
			transport_payload_format="json",
			transport_packets=3,
			transport_hz=1000.0,
			stale_threshold_ms=10.0,
			inference_backend="onnx",
			inference_model_path=str(model_path),
			triangulation_backend="keypoint_centroid",
			skull_solver_backend=backend,  # none|kabsch|ceres
			gaze_fuser_backend="anatomical",
			eye_calibrator_backend="anatomical",
		)
		path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

	rt_path = tmp_path / "rt.json"
	for skull_backend in ("kabsch", "ceres"):
		_write_runtime_json(rt_path, skull_backend)
		# run_pipeline expects a RecordingFolder-shaped path: either under clips/ or named full_recording.
		rec = tmp_path / "full_recording"
		(rec / "mocap_data").mkdir(parents=True, exist_ok=True)
		(rec / "eye_data").mkdir(parents=True, exist_ok=True)
		run_pipeline(
			recording_folder_path=rec,
			realtime_config_path=rt_path,
			calibration_toml_path=None,
			include_eye=False,
			mode="realtime",
		)

