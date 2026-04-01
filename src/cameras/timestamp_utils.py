"""
Timestamp utilities for multi-camera recording.

Handles three concerns:
1. Latching hardware timestamps from the camera array at a point in time.
2. Trimming trailing zeros from the timestamp array after recording.
3. Saving the raw timestamp array and the start/end mappings to disk.

Basler cameras report timestamps in nanoseconds since power-on.
Latching records the current hardware counter value for each camera so that
frame timestamps can be expressed relative to that reference point.
"""

import json
from src.utilities.logging_config import get_logger
import time
from pathlib import Path

import numpy as np
import pypylon.pylon as pylon

from src.cameras.diagnostics.timestamp_mapping import TimestampMapping

logger = get_logger(__name__)


def latch_timestamp_mapping(camera_array: pylon.InstantCameraArray) -> TimestampMapping:
    """
    Latch the hardware timestamp on every camera and return a TimestampMapping.

    Each camera's timestamp counter is latched sequentially. There is a small
    inter-camera latency (nanoseconds to microseconds) between the first and
    last latch — this is later corrected during synchronization.

    Args:
        camera_array: An open InstantCameraArray.

    Returns:
        TimestampMapping with per-camera latch values (nanoseconds since power-on).
    """
    start = time.perf_counter_ns()
    [camera.TimestampLatch.Execute() for camera in camera_array]
    starting_timestamps = {
        camera.GetCameraContext(): camera.TimestampLatchValue.Value
        for camera in camera_array
    }
    mapping = TimestampMapping(camera_timestamps=starting_timestamps)
    elapsed = time.perf_counter_ns() - start
    logger.debug(f"Timestamp latch completed in {elapsed} ns")
    return mapping


def trim_timestamp_zeros(timestamps: np.ndarray) -> np.ndarray:
    """
    Remove trailing zero columns from the timestamp array.

    The timestamp array is pre-allocated with zeros. After recording, columns
    beyond the last written frame are still zero. This trims them.

    Args:
        timestamps: (n_cameras, n_frames) array of per-camera frame timestamps.

    Returns:
        Trimmed (n_cameras, actual_frames) array.
    """
    nonzero = np.nonzero(timestamps)
    if nonzero[1].size == 0:
        return timestamps[:, :0]
    return timestamps[:, : nonzero[1].max() + 1]


def save_timestamps(
    output_path: Path,
    timestamps: np.ndarray,
    starting_mapping: TimestampMapping,
    ending_mapping: TimestampMapping,
) -> None:
    """
    Save the raw timestamp array and the start/end latch mappings to disk.

    Creates two files in output_path:
        timestamps.npy             — (n_cameras, n_frames) array, trailing zeros trimmed
        timestamp_mapping.json     — starting and ending latch mappings

    Args:
        output_path: Directory to write into (must already exist).
        timestamps: (n_cameras, n_frames) timestamp array.
        starting_mapping: TimestampMapping from before recording started.
        ending_mapping: TimestampMapping from after recording stopped.
    """
    trimmed = trim_timestamp_zeros(timestamps)
    np.save(output_path / "timestamps.npy", trimmed)
    logger.info(f"Saved timestamps: shape {trimmed.shape} → {output_path / 'timestamps.npy'}")

    mapping_path = output_path / "timestamp_mapping.json"
    combined = {
        "starting_mapping": starting_mapping.model_dump(),
        "ending_mapping": ending_mapping.model_dump(),
    }
    with open(mapping_path, mode="x") as f:
        json.dump(combined, f, indent=4)
    logger.info(f"Saved timestamp mapping → {mapping_path}")
