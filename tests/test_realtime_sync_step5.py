"""
Step 5 tests for realtime synchronization scaffolding.
"""

from unittest.mock import MagicMock, patch

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


def test_pupil_clock_mapper_builds_from_sample_series() -> None:
    samples = [
        ClockSample(host_monotonic_ns=1_000_000_000, pupil_time_ns=900_000_000),
        ClockSample(host_monotonic_ns=2_000_000_000, pupil_time_ns=1_900_000_000),
        ClockSample(host_monotonic_ns=3_000_000_000, pupil_time_ns=2_900_000_000),
    ]
    mapper = PupilClockMapper.from_sample_series(samples)
    mapped = mapper.pupil_to_host_utc_ns(2_000_000_000)
    assert abs(mapped - 2_100_000_000) <= 1


def test_collect_live_clock_samples_uses_pupil_remote_time() -> None:
    fake_socket = MagicMock()
    fake_socket.recv_string.side_effect = ["1.000000000", "1.001000000"]
    fake_context = MagicMock()
    fake_context.socket.return_value = fake_socket
    fake_zmq = MagicMock()
    fake_zmq.Context.instance.return_value = fake_context
    fake_zmq.REQ = object()
    fake_zmq.RCVTIMEO = 1
    fake_zmq.SNDTIMEO = 2

    with patch.dict("sys.modules", {"zmq": fake_zmq}):
        with patch("src.cameras.synchronization.pupil_clock_sync.time.monotonic_ns") as monotonic_ns:
            monotonic_ns.side_effect = [100, 140, 200, 260]
            samples = PupilClockMapper.from_live_samples(
                endpoint="tcp://127.0.0.1:50020",
                n_samples=2,
                settle_seconds=0.0,
            )
    assert fake_socket.send_string.call_count == 2
    assert isinstance(samples, PupilClockMapper)
