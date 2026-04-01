"""
Shared pytest fixtures for the bs test suite.

Provides:
- minimal_reference_geometry: a simple 3-keypoint ReferenceGeometry
- make_rigid_body_kinematics: factory for RigidBodyKinematics with configurable frames
- fake_clip_dir: a tmp_path-based directory tree matching the expected clip layout
- fake_full_recording_dir: a tmp_path-based full_recording layout for RecordingFolder
- fake_dlc_metadata: writes a skellyclicker_metadata.json with a given iteration
"""

import json
import numpy as np
import pytest

from src.kinematics_core.reference_geometry_model import (
    AxisDefinition,
    AxisType,
    CoordinateFrameDefinition,
    MarkerPosition,
    ReferenceGeometry,
)
from src.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from src.kinematics_core.quaternion_model import Quaternion


# ---------------------------------------------------------------------------
# Reference geometry
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_reference_geometry() -> ReferenceGeometry:
    """Three-keypoint reference geometry (nose, left_eye, right_eye)."""
    return ReferenceGeometry(
        units="mm",
        coordinate_frame=CoordinateFrameDefinition(
            origin_keypoints=["left_eye", "right_eye"],
            x_axis=AxisDefinition(keypoints=["nose"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
        ),
        keypoints={
            "nose":      MarkerPosition(x=18.0, y=0.0, z=0.0),
            "left_eye":  MarkerPosition(x=0.0,  y=12.0, z=0.0),
            "right_eye": MarkerPosition(x=0.0,  y=-12.0, z=0.0),
        },
    )


# ---------------------------------------------------------------------------
# RigidBodyKinematics factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_rigid_body_kinematics(minimal_reference_geometry):
    """
    Factory fixture.  Call with keyword args to override defaults:
        make_rigid_body_kinematics(n_frames=50, name="skull")
    Returns a RigidBodyKinematics with identity orientation and linear position.
    """
    def _factory(
        n_frames: int = 100,
        name: str = "test_body",
        framerate_hz: float = 30.0,
        reference_geometry: ReferenceGeometry | None = None,
    ) -> RigidBodyKinematics:
        ref_geom = reference_geometry or minimal_reference_geometry
        dt = 1.0 / framerate_hz
        timestamps = np.arange(n_frames, dtype=np.float64) * dt
        # Linearly increasing x position, constant y/z
        position_xyz = np.zeros((n_frames, 3), dtype=np.float64)
        position_xyz[:, 0] = np.linspace(0.0, 100.0, n_frames)
        # Identity quaternions [w=1, x=0, y=0, z=0]
        quaternions_wxyz = np.zeros((n_frames, 4), dtype=np.float64)
        quaternions_wxyz[:, 0] = 1.0
        return RigidBodyKinematics.from_pose_arrays(
            name=name,
            reference_geometry=ref_geom,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quaternions_wxyz,
        )
    return _factory


# ---------------------------------------------------------------------------
# Clip directory layout
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_clip_dir(tmp_path, minimal_reference_geometry):
    """
    Builds the minimal directory and file tree expected by ClipPaths /
    run_gaze_pipeline for a single clip.

    Layout:
        tmp_path/
        ├── mocap_data/
        │   ├── output_data/
        │   │   ├── solver_output/
        │   │   │   ├── skull_kinematics.csv          (stub)
        │   │   │   ├── skull_reference_geometry.json
        │   │   │   ├── skull_and_spine_trajectories.csv (stub)
        │   │   │   └── skull_and_spine_topology.json (stub)
        │   │   └── dlc/
        │   │       └── toy_body_3d_xyz.csv (stub)
        │   ├── annotated_videos/
        │   └── synchronized_corrected_videos/
        └── eye_data/
            ├── output_data/
            │   ├── eye0_data.csv (stub)
            │   └── eye1_data.csv (stub)
            └── eye_videos/
    """
    clip = tmp_path

    solver_dir = clip / "mocap_data" / "output_data" / "solver_output"
    solver_dir.mkdir(parents=True)
    dlc_dir = clip / "mocap_data" / "output_data" / "dlc"
    dlc_dir.mkdir(parents=True)
    (clip / "mocap_data" / "annotated_videos").mkdir(parents=True)
    (clip / "mocap_data" / "synchronized_corrected_videos").mkdir(parents=True)

    eye_output_dir = clip / "eye_data" / "output_data"
    eye_output_dir.mkdir(parents=True)
    (clip / "eye_data" / "eye_videos").mkdir(parents=True)

    # Skull reference geometry JSON
    minimal_reference_geometry.to_json_file(solver_dir / "skull_reference_geometry.json")

    # Stub CSVs (just headers + a couple rows)
    _write_stub_kinematics_csv(solver_dir / "skull_kinematics.csv")
    _write_stub_csv(solver_dir / "skull_and_spine_trajectories.csv", ["frame", "x", "y", "z"])
    _write_stub_csv(solver_dir / "skull_and_spine_topology.json", [])
    _write_stub_csv(dlc_dir / "toy_body_3d_xyz.csv", ["frame", "x", "y", "z"])
    _write_stub_eye_csv(eye_output_dir / "eye0_data.csv")
    _write_stub_eye_csv(eye_output_dir / "eye1_data.csv")

    return clip


@pytest.fixture
def fake_full_recording_dir(tmp_path):
    """
    Builds a full_recording directory tree valid for RecordingFolder.from_folder_path.
    The session name contains '420' (not '757') so left_eye='eye1', right_eye='eye0'.
    """
    session_dir = tmp_path / "session_2025-01-01_ferret_420_test"
    recording_dir = session_dir / "full_recording"
    (recording_dir / "mocap_data").mkdir(parents=True)
    (recording_dir / "eye_data").mkdir(parents=True)
    return recording_dir


@pytest.fixture
def fake_dlc_metadata():
    """Returns a helper that writes skellyclicker_metadata.json into a directory."""
    def _write(directory, iteration: int):
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"iteration": iteration, "model": "test_model"}
        (directory / "skellyclicker_metadata.json").write_text(json.dumps(meta))
    return _write


# ---------------------------------------------------------------------------
# Internal helpers (not fixtures)
# ---------------------------------------------------------------------------

def _write_stub_csv(path, columns):
    if not columns:
        path.write_text("{}")
        return
    header = ",".join(columns)
    row = ",".join(["0"] * len(columns))
    path.write_text(f"{header}\n{row}\n")


def _write_stub_kinematics_csv(path):
    """Write a minimal tidy kinematics CSV with position and orientation rows."""
    rows = [
        "frame,timestamp_s,trajectory,component,value,units",
        "0,0.0,position,x,0.0,mm",
        "0,0.0,position,y,0.0,mm",
        "0,0.0,position,z,0.0,mm",
        "0,0.0,orientation,w,1.0,quaternion",
        "0,0.0,orientation,x,0.0,quaternion",
        "0,0.0,orientation,y,0.0,quaternion",
        "0,0.0,orientation,z,0.0,quaternion",
    ]
    path.write_text("\n".join(rows) + "\n")


def _write_stub_eye_csv(path):
    """Write a stub eye DLC output CSV."""
    path.write_text("timestamp_utc,pupil_x,pupil_y,confidence\n0.0,100.0,100.0,0.9\n")
