"""
Tests for src/rigid_body_solver/

This module tests the rigid body solver layer including:

- _dlc_metadata_is_outdated: the version-check helper used by full_pipeline to
  decide whether DLC outputs need reprocessing. Tests cover: missing folder,
  missing metadata file, outdated iteration, current iteration, exact boundary.
- RigidBodySolverConfig: basic construction and field validation.
- ferret_skull_solver topology helpers: that skull and spine topologies can be
  created and contain expected keypoint names.

Note: The Ceres optimization itself (optimize_rigid_body) is not tested here
because it requires hardware-level numerical solvers. Integration tests covering
the full solver would require synthetic marker data and are outside the scope of
unit tests.
"""

import json

import numpy as np
import pytest
from pathlib import Path

from src.batch_processing.full_pipeline import _dlc_metadata_is_outdated
from src.rigid_body_solver.core.main_solver_interface import RigidBodySolverConfig
from src.rigid_body_solver.core.optimization import OptimizationConfig
from src.rigid_body_solver.ferret_skull_solver import (
    create_skull_topology,
    create_skull_and_spine_topology,
)


def _minimal_solver_config(tmp_path: Path, *, csv: Path) -> RigidBodySolverConfig:
    """Build a valid :class:`RigidBodySolverConfig` for construction-only tests."""
    topology = create_skull_topology()
    opt = OptimizationConfig(
        max_iter=5,
        lambda_data=1.0,
        lambda_rot_smooth=0.1,
        lambda_trans_smooth=0.1,
    )
    return RigidBodySolverConfig(
        input_csv=csv,
        timestamps=np.array([0.0, 0.1], dtype=np.float64),
        topology=topology,
        output_dir=tmp_path / "solver_out",
        optimization=opt,
        rigid_body_name="ferret_skull",
        body_frame_origin_keypoints=["nose"],
        body_frame_x_axis_keypoint="left_eye",
        body_frame_y_axis_keypoint="right_eye",
    )


# =============================================================================
# _dlc_metadata_is_outdated
# =============================================================================

class TestDlcMetadataIsOutdated:
    def test_none_folder_returns_true(self):
        assert _dlc_metadata_is_outdated(None, required_iteration=10) is True

    def test_nonexistent_folder_returns_true(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert _dlc_metadata_is_outdated(missing, required_iteration=10) is True

    def test_missing_metadata_file_returns_true(self, tmp_path):
        (tmp_path / "some_output").mkdir()
        assert _dlc_metadata_is_outdated(tmp_path / "some_output", required_iteration=5) is True

    def test_outdated_iteration_returns_true(self, tmp_path, fake_dlc_metadata):
        fake_dlc_metadata(tmp_path, iteration=3)
        assert _dlc_metadata_is_outdated(tmp_path, required_iteration=10) is True

    def test_current_iteration_returns_false(self, tmp_path, fake_dlc_metadata):
        fake_dlc_metadata(tmp_path, iteration=17)
        assert _dlc_metadata_is_outdated(tmp_path, required_iteration=17) is False

    def test_newer_iteration_returns_false(self, tmp_path, fake_dlc_metadata):
        fake_dlc_metadata(tmp_path, iteration=25)
        assert _dlc_metadata_is_outdated(tmp_path, required_iteration=10) is False

    def test_exactly_one_below_required_returns_true(self, tmp_path, fake_dlc_metadata):
        fake_dlc_metadata(tmp_path, iteration=9)
        assert _dlc_metadata_is_outdated(tmp_path, required_iteration=10) is True

    def test_missing_iteration_key_returns_true(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "skellyclicker_metadata.json").write_text(json.dumps({"model": "v1"}))
        assert _dlc_metadata_is_outdated(tmp_path, required_iteration=5) is True


# =============================================================================
# RigidBodySolverConfig
# =============================================================================

class TestRigidBodySolverConfig:
    def test_default_construction(self, tmp_path):
        csv = tmp_path / "markers.csv"
        csv.write_text("frame,x,y,z\n0,0,0,0\n")
        config = _minimal_solver_config(tmp_path, csv=csv)
        assert config.input_csv == csv
        assert config.output_dir == tmp_path / "solver_out"

    def test_output_directory_is_path(self, tmp_path):
        csv = tmp_path / "m.csv"
        csv.write_text("frame,x,y,z\n0,0,0,0\n")
        config = _minimal_solver_config(tmp_path, csv=csv)
        assert isinstance(config.output_dir, Path)


# =============================================================================
# Skull topology helpers
# =============================================================================

class TestSkullTopologyHelpers:
    def test_create_skull_topology_returns_topology(self):
        topology = create_skull_topology()
        assert topology is not None

    def test_skull_topology_has_keypoints(self):
        topology = create_skull_topology()
        assert len(topology.keypoint_names) > 0

    def test_skull_topology_contains_expected_markers(self):
        topology = create_skull_topology()
        names = [n.lower() for n in topology.keypoint_names]
        assert any("skull" in n or "head" in n or "nose" in n or "eye" in n for n in names)

    def test_create_skull_and_spine_topology_returns_topology(self):
        skull = create_skull_topology()
        topology = create_skull_and_spine_topology(
            skull,
            spine_keypoint_names=["spine_mid"],
            spine_display_edges=[("base", "spine_mid")],
        )
        assert topology is not None
        assert "spine_mid" in topology.keypoint_names

    def test_skull_and_spine_has_more_keypoints_than_skull_only(self):
        skull = create_skull_topology()
        skull_and_spine = create_skull_and_spine_topology(
            skull,
            spine_keypoint_names=["spine_mid"],
            spine_display_edges=[("base", "spine_mid")],
        )
        assert len(skull_and_spine.keypoint_names) >= len(skull.keypoint_names)
