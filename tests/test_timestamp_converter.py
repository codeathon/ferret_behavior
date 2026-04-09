"""Regression tests for timestamp conversion edge cases."""

import numpy as np

from src.cameras.synchronization.timestamp_converter import TimestampConverter


def test_get_closest_pupil_frame_uses_dict_backed_basler_timestamps():
    """Ensure dict-backed Basler UTC timestamps are handled without ndarray slicing."""
    converter = TimestampConverter.__new__(TimestampConverter)

    # Simulate per-camera synchronized Basler UTC arrays stored in a dict.
    converter.synched_basler_timestamps_utc = {
        "0": np.array([100, 200, 300], dtype=np.int64),
        "1": np.array([110, 210, 310], dtype=np.int64),
    }
    converter.pupil_eye0_timestamps_utc = np.array([150, 205, 260], dtype=np.int64)
    converter.pupil_eye1_timestamps_utc = np.array([150, 208, 260], dtype=np.int64)
    converter.include_pupil_world = False

    eye0_frame, eye1_frame, world_frame = converter.get_closest_pupil_frame_to_basler_frame(1)

    # Median Basler UTC at frame 1 is 205, so nearest eye frames should be index 1.
    assert eye0_frame == 1
    assert eye1_frame == 1
    assert world_frame is None
