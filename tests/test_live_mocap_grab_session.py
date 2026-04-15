"""Tests for Basler grab -> live mocap publish session (mocked hardware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cameras.synchronization.realtime_sync import BaslerFrame, BaslerFrameSet
from src.ferret_gaze.realtime.live_mocap_grab_session import run_live_mocap_grab_n_frames_publish
from src.ferret_gaze.realtime.per_frame_compute import StubInferenceRuntime, StubTriangulator
from src.ferret_gaze.realtime.publisher import NoOpRealtimePublisher


def test_grab_n_frames_publish_invokes_grab_with_sink_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeMCR:
        """Minimal stand-in; real path is integration-tested on hardware."""

        camera_array = object()
        devices: list[object] = []
        output_path = tmp_path

        def __init__(self, output_path: Path, nir_only: bool, fps: float) -> None:
            captured["init"] = (output_path, nir_only, fps)
            self.output_path = output_path

        def open_camera_array(self) -> None:
            pass

        def set_max_num_buffer(self, n: int) -> None:
            pass

        def set_fps(self, f: float) -> None:
            pass

        def set_image_resolution(self, binning_factor: int) -> None:
            pass

        def set_hardware_triggering(self, hardware_triggering: bool) -> None:
            pass

        def camera_information(self) -> None:
            pass

        def create_video_writers_ffmpeg(self) -> None:
            pass

        def close_camera_array(self) -> None:
            pass

        def grab_n_frames(self, n: int, frameset_sink: object | None = None) -> None:
            captured["n_frames"] = n
            captured["sink"] = frameset_sink
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            bset = BaslerFrameSet(
                anchor_utc_ns=1000,
                frames_by_camera={
                    0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1000, payload=img),
                },
            )
            if frameset_sink is not None:
                frameset_sink(bset)

    monkeypatch.setattr(
        "src.ferret_gaze.realtime.live_mocap_grab_session.MultiCameraRecording",
        _FakeMCR,
    )
    monkeypatch.setattr(
        "src.ferret_gaze.realtime.live_mocap_grab_session.configure_all_cameras",
        lambda **kwargs: None,
    )

    pub = NoOpRealtimePublisher()
    summary = run_live_mocap_grab_n_frames_publish(
        output_path=tmp_path,
        nir_only=False,
        fps=60.0,
        binning_factor=2,
        hardware_triggering=False,
        n_frames=3,
        publisher=pub,
        inference_runtime=StubInferenceRuntime(),
        triangulator=StubTriangulator(),
        stale_threshold_ms=80.0,
        wire_queue_size=8,
    )
    assert captured["n_frames"] == 3
    assert captured["sink"] is not None
    assert summary is not None
    assert summary.packet_count == 1


def test_grab_n_frames_publish_rejects_zero_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_frames"):
        run_live_mocap_grab_n_frames_publish(
            output_path=tmp_path,
            nir_only=False,
            fps=60.0,
            binning_factor=2,
            hardware_triggering=False,
            n_frames=0,
            publisher=NoOpRealtimePublisher(),
            inference_runtime=StubInferenceRuntime(),
            triangulator=StubTriangulator(),
            stale_threshold_ms=80.0,
        )
