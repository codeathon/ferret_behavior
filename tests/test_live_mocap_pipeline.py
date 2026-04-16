"""Tests for live mocap orchestration (frame sets -> compute -> publish)."""

from __future__ import annotations

import numpy as np
import pytest

from src.cameras.synchronization.realtime_sync import BaslerFrame, BaslerFrameSet
from src.ferret_gaze.realtime.live_frame_set import LiveMocapFrameSet
from src.ferret_gaze.realtime.live_mocap_pipeline import (
    basler_frameset_to_live_mocap_frame_set,
    build_synthetic_live_mocap_frame_sets,
    gaze_packet_from_live_mocap_frame_set,
    process_live_mocap_tick,
    run_live_mocap_compute_publish_session,
)
from src.ferret_gaze.realtime.per_frame_compute import (
    StubGazeFuser,
    StubInferenceRuntime,
    StubRollingEyeCalibrator,
    StubTriangulator,
)
from src.ferret_gaze.realtime.publisher import NoOpRealtimePublisher


def test_gaze_packet_from_live_frame_set_matches_seq_and_anchor() -> None:
    fs = LiveMocapFrameSet(
        seq=5,
        anchor_utc_ns=9_000_000_000,
        images_bgr={0: np.zeros((4, 4, 3), dtype=np.uint8)},
    )
    packet = gaze_packet_from_live_mocap_frame_set(fs)
    assert packet.seq == 5
    assert packet.capture_utc_ns == 9_000_000_000


def test_basler_frameset_to_live_requires_bgr_payload() -> None:
    bset = BaslerFrameSet(
        anchor_utc_ns=1,
        frames_by_camera={
            0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1, payload=None),
        },
    )
    assert basler_frameset_to_live_mocap_frame_set(bset, seq=0) is None


def test_basler_frameset_to_live_maps_bgr() -> None:
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[0, 0, 1] = 200
    bset = BaslerFrameSet(
        anchor_utc_ns=42,
        frames_by_camera={
            1: BaslerFrame(camera_id=1, frame_index=0, capture_utc_ns=42, payload=img),
        },
    )
    live = basler_frameset_to_live_mocap_frame_set(bset, seq=3)
    assert live is not None
    assert live.seq == 3
    assert live.anchor_utc_ns == 42
    assert np.array_equal(live.images_bgr[1], img)


def test_build_synthetic_live_mocap_frame_sets_shape() -> None:
    fss = build_synthetic_live_mocap_frame_sets(2, n_cams=3, height=10, width=12, anchor_base_ns=100)
    assert len(fss) == 2
    assert fss[0].images_bgr[0].shape == (10, 12, 3)
    assert fss[1].anchor_utc_ns == 100 + 1_000_000


def test_build_synthetic_live_mocap_frame_sets_dummy_pupil_eyes() -> None:
    fss = build_synthetic_live_mocap_frame_sets(1, attach_dummy_pupil_eyes=True, anchor_base_ns=9_000)
    assert fss[0].eye0_bgr is not None
    assert fss[0].eye1_bgr is not None
    assert fss[0].pupil_eye0_utc_ns == 9_000
    assert fss[0].pupil_eye0_stale is False


def test_build_synthetic_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        build_synthetic_live_mocap_frame_sets(0)


def test_run_live_mocap_compute_publish_session_smoke() -> None:
    fss = build_synthetic_live_mocap_frame_sets(2, n_cams=2, height=8, width=8)
    publisher = NoOpRealtimePublisher()
    summary = run_live_mocap_compute_publish_session(
        frame_sets=fss,
        publisher=publisher,
        inference_runtime=StubInferenceRuntime(),
        triangulator=StubTriangulator(),
        hz=10_000.0,
        stale_threshold_ms=80.0,
    )
    assert summary.packet_count == 2


def test_process_live_mocap_tick_returns_fused_packet() -> None:
    fs = build_synthetic_live_mocap_frame_sets(1, n_cams=1, height=4, width=4)[0]
    fused = process_live_mocap_tick(
        fs,
        inference_runtime=StubInferenceRuntime(),
        triangulator=StubTriangulator(),
        calibrator=StubRollingEyeCalibrator(),
        fuser=StubGazeFuser(),
    )
    assert fused.seq == fs.seq


def test_run_live_mocap_rejects_non_positive_hz() -> None:
    with pytest.raises(ValueError, match="hz"):
        run_live_mocap_compute_publish_session(
            frame_sets=build_synthetic_live_mocap_frame_sets(1),
            publisher=NoOpRealtimePublisher(),
            inference_runtime=StubInferenceRuntime(),
            triangulator=StubTriangulator(),
            hz=0.0,
            stale_threshold_ms=80.0,
        )
