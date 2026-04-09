"""
Step 5 tests for realtime synchronization scaffolding.
"""

from src.cameras.synchronization.pupil_clock_sync import ClockSample, PupilClockMapper
from src.cameras.synchronization.realtime_sync import (
    BaslerFrame,
    BaslerFrameSetCombiner,
    associate_pupil_frames_to_basler,
    nearest_timestamp_index,
)


def test_basler_combiner_emits_frameset_when_all_cameras_available() -> None:
    combiner = BaslerFrameSetCombiner(camera_ids=[0, 1], ring_size=8, tolerance_ns=2_000_000)
    assert combiner.ingest(BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=100_000_000)) is None
    frameset = combiner.ingest(BaslerFrame(camera_id=1, frame_index=1, capture_utc_ns=101_000_000))
    assert frameset is not None
    assert sorted(frameset.frames_by_camera.keys()) == [0, 1]


def test_basler_combiner_respects_tolerance() -> None:
    combiner = BaslerFrameSetCombiner(camera_ids=[0, 1], ring_size=8, tolerance_ns=500_000)
    assert combiner.ingest(BaslerFrame(camera_id=0, frame_index=1, capture_utc_ns=100_000_000)) is None
    frameset = combiner.ingest(BaslerFrame(camera_id=1, frame_index=1, capture_utc_ns=102_000_000))
    assert frameset is None


def test_nearest_association_helpers() -> None:
    values = [100, 210, 330, 460]
    assert nearest_timestamp_index(values, 340) == 2
    eye0, eye1 = associate_pupil_frames_to_basler(
        basler_utc_ns=350,
        pupil_eye0_timestamps_utc_ns=[90, 220, 360],
        pupil_eye1_timestamps_utc_ns=[95, 240, 355],
    )
    assert eye0 == 2
    assert eye1 == 2


def test_pupil_clock_mapper_round_trip_after_sample_update() -> None:
    mapper = PupilClockMapper()
    mapper.update_from_samples(
        first=ClockSample(host_monotonic_ns=1_000_000_000, pupil_time_ns=900_000_000),
        second=ClockSample(host_monotonic_ns=2_000_000_000, pupil_time_ns=1_900_000_000),
    )
    mapped = mapper.pupil_to_host_utc_ns(1_400_000_000)
    recovered = mapper.host_to_pupil_time_ns(mapped)
    assert abs(recovered - 1_400_000_000) <= 1
