"""
Shared tidy-format Polars DataFrame helpers for kinematics serialization.

Used by kinematics_core and ferret_gaze serialization modules.
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray


def build_vector_chunk(
    frame_indices: NDArray[np.int64],
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    trajectory_name: str,
    component_names: list[str],
    units: str,
) -> pl.DataFrame:
    """
    Build a tidy DataFrame chunk for a vector quantity using vectorized ops.

    Each (frame, component) pair becomes one row.

    Args:
        frame_indices:   (N,) array of frame indices
        timestamps:      (N,) array of timestamps in seconds
        values:          (N, C) array where C == len(component_names)
        trajectory_name: Label for the trajectory column
        component_names: List of component labels, length C
        units:           Unit string (e.g. "mm", "rad_s", "quaternion")

    Returns:
        Tidy polars DataFrame with N * C rows and columns:
        [frame, timestamp_s, trajectory, component, value, units]
    """
    n_frames = len(frame_indices)
    n_components = len(component_names)

    # [0,0,0, 1,1,1, ...] — repeat each frame index once per component
    repeated_frames = np.repeat(frame_indices, n_components)
    repeated_timestamps = np.repeat(timestamps, n_components)

    # ["x","y","z", "x","y","z", ...] — tile names once per frame
    tiled_components = np.tile(component_names, n_frames)

    # flatten row-major: all components for frame 0, then frame 1, etc.
    flattened_values = values.ravel()

    return (
        pl.DataFrame({
            "frame": repeated_frames,
            "timestamp_s": repeated_timestamps,
            "component": tiled_components,
            "value": flattened_values,
        })
        .with_columns(
            pl.lit(trajectory_name).alias("trajectory").cast(pl.Categorical),
            pl.col("component").cast(pl.Categorical),
            pl.lit(units).alias("units").cast(pl.Categorical),
        )
        .select(["frame", "timestamp_s", "trajectory", "component", "value", "units"])
    )


def extract_timestamps_from_tidy_df(df: pl.DataFrame) -> NDArray[np.float64]:
    """
    Extract unique per-frame timestamps from a tidy kinematics DataFrame.

    Args:
        df: Tidy-format DataFrame with at least [frame, timestamp_s] columns

    Returns:
        (N,) float64 array of timestamps sorted by frame index
    """
    return (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )
