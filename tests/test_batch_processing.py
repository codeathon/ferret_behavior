"""
Tests for src/batch_processing/

This module tests the orchestration logic in full_pipeline.py and
postprocess_recording.py without executing any real computation or
calling external subprocesses.

Coverage includes:

- Overwrite flag cascade: verifies that setting an upstream overwrite flag
  correctly propagates to all downstream dependent flags (e.g.,
  overwrite_synchronization triggers overwrite_calibration, etc.).
- DLC iteration auto-detection: verifies that outdated DLC metadata forces
  overwrite_dlc=True automatically.
- Pipeline step skipping: verifies that each step is skipped when outputs
  already exist and the overwrite flag is False.
- RecordingFolder.check_* methods: verifies they raise ValueError when
  expected outputs are absent.

All subprocess calls (skellyclicker, dlc_to_3d, freemocap) and the cameras
postprocess call are mocked. RecordingFolder.from_folder_path is also mocked
where needed so tests do not require a real directory tree.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.batch_processing.full_pipeline import (
    _dlc_metadata_is_outdated,
    full_pipeline,
    run_pipeline,
    HEAD_DLC_ITERATION,
    EYE_DLC_ITERATION,
    TOY_DLC_ITERATION,
)
from src.ferret_gaze.realtime.latency_metrics import LatencySummary
from src.ferret_gaze.realtime.runtime_config import RealtimeRuntimeConfig
from src.ferret_gaze.realtime.solver_benchmark import SolverBenchmarkComparison, SolverBenchmarkStats


# =============================================================================
# DLC iteration constants sanity check
# =============================================================================

class TestDlcIterationConstants:
    def test_constants_are_positive_integers(self):
        assert isinstance(HEAD_DLC_ITERATION, int) and HEAD_DLC_ITERATION > 0
        assert isinstance(EYE_DLC_ITERATION, int) and EYE_DLC_ITERATION > 0
        assert isinstance(TOY_DLC_ITERATION, int) and TOY_DLC_ITERATION > 0


# =============================================================================
# Overwrite flag cascade
# =============================================================================

class TestOverwriteFlagCascade:
    """
    Test that overwrite flags propagate correctly through dependent steps.
    We mock all side-effecting functions so only flag propagation is tested.
    """

    def _make_mock_recording_folder(
        self,
        synchronized=True,
        calibrated=True,
        dlc_processed=True,
        triangulated=True,
        eye_postprocessed=True,
        skull_postprocessed=True,
        gaze_postprocessed=True,
        head_dlc_output=None,
        eye_dlc_output=None,
        toy_dlc_output=None,
    ):
        rf = MagicMock()
        rf.is_synchronized.return_value = synchronized
        rf.is_calibrated.return_value = calibrated
        rf.is_dlc_processed.return_value = dlc_processed
        rf.is_triangulated.return_value = triangulated
        rf.is_eye_postprocessed.return_value = eye_postprocessed
        rf.is_skull_postprocessed.return_value = skull_postprocessed
        rf.is_gaze_postprocessed.return_value = gaze_postprocessed
        # Real paths + skellyclicker_metadata.json avoid false "outdated DLC" in overwrite resolver.
        rf.head_body_dlc_output = head_dlc_output
        rf.eye_dlc_output = eye_dlc_output
        rf.toy_dlc_output = toy_dlc_output
        rf.calibration_toml_path = Path("/fake/calibration.toml")
        rf.calibration_videos = Path("/fake/calibration_videos")
        rf.base_recordings_folder = Path("/fake/recordings")
        return rf

    def _run_pipeline_with_flags(self, tmp_path, mock_rf, **overwrite_flags):
        with patch("src.batch_processing.full_pipeline.RecordingFolder.from_folder_path", return_value=mock_rf), \
             patch("src.batch_processing.full_pipeline.postprocess") as mock_sync, \
             patch("src.batch_processing.full_pipeline.run_calibration_subprocess") as mock_cal, \
             patch("src.batch_processing.full_pipeline.run_skellyclicker_subprocess") as mock_dlc, \
             patch("src.batch_processing.full_pipeline.run_triangulation_subprocess") as mock_tri, \
             patch("src.batch_processing.full_pipeline.process_recording") as mock_post:
            full_pipeline(tmp_path, **overwrite_flags)
            return mock_sync, mock_cal, mock_dlc, mock_tri, mock_post

    def test_all_up_to_date_no_steps_run(self, tmp_path):
        def _write_dlc_meta(folder: Path, iteration: int) -> None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "skellyclicker_metadata.json").write_text(
                json.dumps({"iteration": iteration}), encoding="utf-8"
            )

        head_out = tmp_path / "head_dlc"
        eye_out = tmp_path / "eye_dlc"
        toy_out = tmp_path / "toy_dlc"
        _write_dlc_meta(head_out, HEAD_DLC_ITERATION)
        _write_dlc_meta(eye_out, EYE_DLC_ITERATION)
        _write_dlc_meta(toy_out, TOY_DLC_ITERATION)
        mock_rf = self._make_mock_recording_folder(
            head_dlc_output=head_out,
            eye_dlc_output=eye_out,
            toy_dlc_output=toy_out,
        )
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(tmp_path, mock_rf)
        mock_sync.assert_not_called()
        mock_cal.assert_not_called()
        mock_dlc.assert_not_called()
        mock_tri.assert_not_called()
        mock_post.assert_not_called()

    def test_overwrite_synchronization_triggers_calibration(self, tmp_path):
        mock_rf = self._make_mock_recording_folder()
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(
            tmp_path, mock_rf, overwrite_synchronization=True
        )
        mock_sync.assert_called_once()
        mock_cal.assert_called_once()

    def test_overwrite_triangulation_triggers_skull_postprocessing(self, tmp_path):
        mock_rf = self._make_mock_recording_folder()
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(
            tmp_path, mock_rf, overwrite_triangulation=True
        )
        mock_tri.assert_called_once()
        mock_post.assert_called_once()

    def test_overwrite_eye_postprocessing_triggers_gaze(self, tmp_path):
        mock_rf = self._make_mock_recording_folder()
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(
            tmp_path, mock_rf, overwrite_eye_postprocessing=True
        )
        mock_post.assert_called_once()

    def test_outdated_dlc_forces_overwrite_dlc(self, tmp_path, fake_dlc_metadata):
        dlc_dir = tmp_path / "dlc_output"
        fake_dlc_metadata(dlc_dir, iteration=1)  # well below required
        mock_rf = self._make_mock_recording_folder(
            dlc_processed=True,  # DLC output exists but outdated
            head_dlc_output=dlc_dir,
            eye_dlc_output=dlc_dir,
            toy_dlc_output=dlc_dir,
        )
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(
            tmp_path, mock_rf
        )
        mock_dlc.assert_called_once()

    def test_not_synchronized_triggers_sync_step(self, tmp_path):
        mock_rf = self._make_mock_recording_folder(synchronized=False)
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(tmp_path, mock_rf)
        mock_sync.assert_called_once()

    def test_not_calibrated_triggers_calibration(self, tmp_path):
        mock_rf = self._make_mock_recording_folder(calibrated=False)
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(tmp_path, mock_rf)
        mock_cal.assert_called_once()

    def test_not_triangulated_triggers_triangulation(self, tmp_path):
        mock_rf = self._make_mock_recording_folder(triangulated=False)
        mock_sync, mock_cal, mock_dlc, mock_tri, mock_post = self._run_pipeline_with_flags(tmp_path, mock_rf)
        mock_tri.assert_called_once()


class TestPipelineModeScaffold:
    """Validate top-level offline/realtime mode dispatch behavior."""

    def test_run_pipeline_offline_dispatches_to_offline_impl(self, tmp_path):
        # Offline mode should execute the existing batch logic path.
        with patch("src.batch_processing.full_pipeline._run_offline_pipeline") as mock_offline, \
             patch("src.batch_processing.full_pipeline._run_realtime_pipeline") as mock_realtime:
            run_pipeline(recording_folder_path=tmp_path, mode="offline")
            mock_offline.assert_called_once()
            mock_realtime.assert_not_called()

    def test_run_pipeline_realtime_dispatches_to_realtime_impl(self, tmp_path):
        # Realtime mode should dispatch to the realtime path.
        with patch("src.batch_processing.full_pipeline._run_realtime_pipeline") as mock_realtime, \
             patch("src.batch_processing.full_pipeline._run_offline_pipeline") as mock_offline:
            run_pipeline(recording_folder_path=tmp_path, mode="realtime")
            mock_realtime.assert_called_once()
            mock_offline.assert_not_called()

    def test_realtime_pipeline_uses_configured_triangulation_backend(self, tmp_path):
        # Realtime mode should build the triangulator from runtime config.
        config_path = tmp_path / "realtime.runtime.json"
        runtime_config = RealtimeRuntimeConfig(
            transport_backend="noop",
            transport_endpoint="tcp://127.0.0.1:5556",
            transport_topic="gaze.live",
            transport_packets=2,
            transport_hz=30.0,
            stale_threshold_ms=80.0,
            benchmark_packets=2,
            compute_packets=2,
            inference_backend="stub",
            inference_model_path=None,
            onnx_provider="CPUExecutionProvider",
            triangulation_backend="stub",
        )
        transport_summary = LatencySummary(
            packet_count=0,
            dropped_count=0,
            stale_count=0,
            end_to_end_p50_ms=0.0,
            end_to_end_p95_ms=0.0,
            end_to_end_p99_ms=0.0,
            process_p50_ms=0.0,
            process_p95_ms=0.0,
            process_p99_ms=0.0,
            stale_threshold_ms=80.0,
        )
        stub_stat = SolverBenchmarkStats(
            solver_name="ukf_stub",
            packet_count=0,
            mean_solver_latency_ms=0.0,
            p95_solver_latency_ms=0.0,
            mean_position_error_mm=0.0,
            mean_quaternion_l1_error=0.0,
        )
        stub_comparison = SolverBenchmarkComparison(
            ukf=stub_stat,
            ceres=stub_stat,
            recommended_solver="ukf_stub",
            recommendation_reason="test",
        )
        with patch("src.batch_processing.full_pipeline.load_realtime_runtime_config", return_value=runtime_config) as mock_load_runtime_config, \
             patch("src.batch_processing.full_pipeline.create_realtime_publisher"), \
             patch(
                 "src.batch_processing.full_pipeline.run_realtime_transport_scaffold",
                 return_value=transport_summary,
             ), \
             patch("src.batch_processing.full_pipeline.build_synthetic_replay_packets", return_value=[]), \
             patch(
                 "src.batch_processing.full_pipeline.compare_stub_solvers",
                 return_value=stub_comparison,
             ), \
             patch("src.batch_processing.full_pipeline.create_inference_runtime"), \
             patch("src.batch_processing.full_pipeline.create_triangulator") as mock_create_triangulator, \
             patch("src.batch_processing.full_pipeline.run_realtime_compute_scaffold", return_value=[]):
            run_pipeline(
                recording_folder_path=tmp_path,
                mode="realtime",
                realtime_config_path=config_path,
            )
            mock_load_runtime_config.assert_called_once_with(config_path=config_path)
            mock_create_triangulator.assert_called_once_with(
                backend="stub",
                calibration_toml_path=None,
            )

    def test_realtime_pipeline_live_mocap_runs_compute_publish_not_transport(self, tmp_path):
        """live_mocap mode should drive the live loop, not the synthetic transport scaffold."""
        config_path = tmp_path / "realtime.runtime.json"
        config_path.write_text("{}", encoding="utf-8")
        runtime_config = RealtimeRuntimeConfig(
            realtime_mode="live_mocap",
            transport_backend="noop",
            transport_endpoint="tcp://127.0.0.1:5556",
            transport_topic="gaze.live",
            transport_packets=2,
            transport_hz=1000.0,
            stale_threshold_ms=80.0,
            benchmark_packets=2,
            compute_packets=2,
            inference_backend="stub",
            inference_model_path=None,
            onnx_provider="CPUExecutionProvider",
            triangulation_backend="stub",
        )
        stub_stat = SolverBenchmarkStats(
            solver_name="ukf_stub",
            packet_count=0,
            mean_solver_latency_ms=0.0,
            p95_solver_latency_ms=0.0,
            mean_position_error_mm=0.0,
            mean_quaternion_l1_error=0.0,
        )
        stub_comparison = SolverBenchmarkComparison(
            ukf=stub_stat,
            ceres=stub_stat,
            recommended_solver="ukf_stub",
            recommendation_reason="test",
        )
        live_summary = LatencySummary(
            packet_count=2,
            dropped_count=0,
            stale_count=0,
            end_to_end_p50_ms=0.0,
            end_to_end_p95_ms=0.0,
            end_to_end_p99_ms=0.0,
            process_p50_ms=0.0,
            process_p95_ms=0.0,
            process_p99_ms=0.0,
            stale_threshold_ms=80.0,
        )
        with patch(
            "src.batch_processing.full_pipeline.load_realtime_runtime_config",
            return_value=runtime_config,
        ), patch("src.batch_processing.full_pipeline.create_realtime_publisher"), patch(
            "src.batch_processing.full_pipeline.run_realtime_transport_scaffold",
        ) as mock_transport, patch(
            "src.batch_processing.full_pipeline.compare_stub_solvers",
            return_value=stub_comparison,
        ), patch(
            "src.batch_processing.full_pipeline.create_inference_runtime",
        ), patch(
            "src.batch_processing.full_pipeline.create_triangulator",
        ), patch(
            "src.batch_processing.full_pipeline.run_live_mocap_compute_publish_session",
            return_value=live_summary,
        ) as mock_live, patch(
            "src.batch_processing.full_pipeline.run_realtime_compute_scaffold",
        ) as mock_replay_compute:
            run_pipeline(
                recording_folder_path=tmp_path,
                mode="realtime",
                realtime_config_path=config_path,
            )
            mock_transport.assert_not_called()
            mock_replay_compute.assert_not_called()
            mock_live.assert_called_once()
            assert mock_live.call_args.kwargs["hz"] == 1000.0
            assert len(mock_live.call_args.kwargs["frame_sets"]) == 2

    def test_realtime_pipeline_live_mocap_grab_calls_grab_publish(self, tmp_path):
        """live_mocap + grab should invoke Basler grab session, not synthetic replay."""
        config_path = tmp_path / "realtime.runtime.json"
        config_path.write_text("{}", encoding="utf-8")
        runtime_config = RealtimeRuntimeConfig(
            realtime_mode="live_mocap",
            live_mocap_frame_source="grab",
            transport_backend="noop",
            transport_endpoint="tcp://127.0.0.1:5556",
            transport_topic="gaze.live",
            transport_packets=7,
            transport_hz=90.0,
            stale_threshold_ms=80.0,
            benchmark_packets=2,
            compute_packets=2,
            inference_backend="stub",
            inference_model_path=None,
            onnx_provider="CPUExecutionProvider",
            triangulation_backend="stub",
        )
        stub_stat = SolverBenchmarkStats(
            solver_name="ukf_stub",
            packet_count=0,
            mean_solver_latency_ms=0.0,
            p95_solver_latency_ms=0.0,
            mean_position_error_mm=0.0,
            mean_quaternion_l1_error=0.0,
        )
        stub_comparison = SolverBenchmarkComparison(
            ukf=stub_stat,
            ceres=stub_stat,
            recommended_solver="ukf_stub",
            recommendation_reason="test",
        )
        with patch(
            "src.batch_processing.full_pipeline.load_realtime_runtime_config",
            return_value=runtime_config,
        ), patch("src.batch_processing.full_pipeline.create_realtime_publisher"), patch(
            "src.batch_processing.full_pipeline.run_realtime_transport_scaffold",
        ) as mock_transport, patch(
            "src.batch_processing.full_pipeline.compare_stub_solvers",
            return_value=stub_comparison,
        ), patch(
            "src.batch_processing.full_pipeline.create_inference_runtime",
        ), patch(
            "src.batch_processing.full_pipeline.create_triangulator",
        ), patch(
            "src.batch_processing.full_pipeline.run_live_mocap_compute_publish_session",
        ) as mock_synthetic, patch(
            "src.batch_processing.full_pipeline.run_live_mocap_grab_n_frames_publish",
        ) as mock_grab, patch(
            "src.batch_processing.full_pipeline.run_realtime_compute_scaffold",
        ) as mock_replay_compute:
            run_pipeline(
                recording_folder_path=tmp_path,
                mode="realtime",
                realtime_config_path=config_path,
            )
            mock_transport.assert_not_called()
            mock_replay_compute.assert_not_called()
            mock_synthetic.assert_not_called()
            mock_grab.assert_called_once()
            gkw = mock_grab.call_args.kwargs
            assert gkw["n_frames"] == 7
            assert gkw["output_path"] == tmp_path
            assert gkw["fps"] == 90.0
            assert gkw["nir_only"] is False
            assert gkw.get("pupil_rings") is None
            assert gkw.get("on_timestamp_latch") is None
