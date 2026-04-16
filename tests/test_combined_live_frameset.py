"""Tests for live Basler + Pupil combined frame association and clock bridge."""

from __future__ import annotations

import numpy as np

from src.cameras.diagnostics.timestamp_mapping import TimestampMapping
from src.cameras.synchronization.pupil_clock_sync import ClockSample, PupilClockMapper
from src.cameras.synchronization.pupil_dual_eye_rings import PupilDualEyeRings
from src.cameras.synchronization.pupil_live_queue_ingest import PupilWallUtcQueueIngestThread
from src.cameras.synchronization.realtime_sync import (
    BaslerFrame,
    BaslerFrameSet,
    associate_pupil_frames_to_basler,
    associate_pupil_frames_to_basler_sorted,
    nearest_sorted_timestamp_index,
    nearest_timestamp_index,
)
from src.cameras.synchronization.utc_clock_bridge import WallUtcFromPupilTime
from src.ferret_gaze.realtime.live_mocap_pipeline import basler_frameset_to_live_mocap_frame_set


def test_nearest_sorted_matches_linear_scan_on_sorted_lists() -> None:
    rng = np.random.default_rng(0)
    for _ in range(32):
        n = rng.integers(5, 40)
        raw = sorted(int(x) for x in rng.integers(0, 1_000_000_000, size=n))
        target = int(rng.integers(0, 1_000_000_000))
        i_sorted = nearest_sorted_timestamp_index(raw, target)
        i_brute = nearest_timestamp_index(list(raw), target)
        assert i_sorted == i_brute


def test_associate_pupil_sorted_matches_unsorted_helper() -> None:
    e0 = [100, 200, 300, 400, 500]
    e1 = [150, 250, 350, 450, 550]
    t = 330
    a0, a1 = associate_pupil_frames_to_basler(t, list(e0), list(e1))
    b0, b1 = associate_pupil_frames_to_basler_sorted(t, e0, e1)
    assert (a0, a1) == (b0, b1)


def test_wall_utc_from_pupil_time_aligns_at_latch_monotonic() -> None:
    latch = TimestampMapping(
        camera_timestamps={0: 0},
        utc_time_ns=1_700_000_000_000_000_000,
        perf_counter_ns=1_000,
        monotonic_ns=50_000,
    )
    samples = [
        ClockSample(host_monotonic_ns=100_000, pupil_time_ns=100_000),
        ClockSample(host_monotonic_ns=200_000, pupil_time_ns=200_000),
    ]
    mapper = PupilClockMapper.from_sample_series(samples)
    bridge = WallUtcFromPupilTime(latch=latch, mapper=mapper)
    wall_at_latch_mono = bridge.pupil_time_ns_to_wall_utc_ns(latch.monotonic_ns)
    assert wall_at_latch_mono == latch.utc_time_ns


def test_pupil_dual_eye_rings_nearest_and_stale() -> None:
    rings = PupilDualEyeRings(maxlen_per_eye=64)
    anchor = 1_000_000
    stale_limit = 10_000
    img0 = np.zeros((4, 4, 3), dtype=np.uint8)
    img1 = np.ones((4, 4, 3), dtype=np.uint8)
    rings.push_eye0_wall(anchor - 5, img0)
    rings.push_eye1_wall(anchor + 100_000, img1)
    a0, a1 = rings.associate(anchor, stale_limit)
    assert a0.stale is False
    assert a0.delta_ns == 5
    assert np.array_equal(a0.image, img0)
    assert a1.stale is True
    assert a1.delta_ns == 100_000


def test_basler_frameset_to_live_mocap_frame_set_with_pupil_rings() -> None:
    rings = PupilDualEyeRings(maxlen_per_eye=32)
    anchor = 5_000_000
    img_cam = np.zeros((2, 2, 3), dtype=np.uint8)
    img_e0 = np.full((2, 2, 3), 3, dtype=np.uint8)
    img_e1 = np.full((2, 2, 3), 7, dtype=np.uint8)
    rings.push_eye0_wall(anchor, img_e0)
    rings.push_eye1_wall(anchor + 1, img_e1)
    basler = BaslerFrameSet(
        anchor_utc_ns=anchor,
        frames_by_camera={
            0: BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=anchor, payload=img_cam),
            1: BaslerFrame(camera_id=1, frame_index=1, capture_utc_ns=anchor, payload=img_cam),
        },
    )
    live = basler_frameset_to_live_mocap_frame_set(
        basler,
        seq=0,
        pupil_rings=rings,
        pupil_stale_max_delta_ns=50_000,
    )
    assert live is not None
    assert live.pupil_eye0_stale is False
    assert live.pupil_eye1_stale is False
    assert np.array_equal(live.eye0_bgr, img_e0)
    assert np.array_equal(live.eye1_bgr, img_e1)


def test_pupil_wall_utc_queue_ingest_thread() -> None:
    rings = PupilDualEyeRings(maxlen_per_eye=32)
    worker = PupilWallUtcQueueIngestThread(rings, max_queue=16)
    worker.start()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    worker.queue.put((0, 123, img))
    worker.queue.put((1, 124, img))
    worker.stop(join_timeout_s=2.0)
    a0, a1 = rings.associate(123, stale_max_delta_ns=1_000_000)
    assert a0.stale is False
    assert a1.stale is False
