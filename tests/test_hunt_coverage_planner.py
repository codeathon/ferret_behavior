"""Unit tests for hunt coverage planning (no Zaber hardware)."""

from __future__ import annotations

from src.hunt.config import HuntConfig, WorkspaceMm
from src.hunt.coverage_planner import build_coverage_plan, extend_plan_for_min_duration
from src.hunt.runner import build_full_plan, estimate_plan_duration_s
from src.hunt.workspace import clip_xy, corner_points, grid_points
from src.hunt.zaber_client import FakeGantry, PoseMm


def _cfg(**overrides) -> HuntConfig:
	base = dict(
		port="/dev/null",
		baud_rate=115200,
		device_address=1,
		axis_x=1,
		axis_y=2,
		axis_x2=0,
		lockstep_group=1,
		workspace=WorkspaceMm(0, 1000, 0, 1000),
		coverage_grid_mm=250.0,
		min_run_s=60.0,
		speeds_mps=(0.1, 0.3),
		distances_mm=(50.0, 200.0),
		bearings_deg=(0.0, 90.0, 180.0, 270.0),
		corner_margin_mm=50.0,
		flee_distances_mm=(200.0, 400.0),
		hunt_rearm_s=1.5,
		max_accel_mps2=2.0,
		settle_tol_mm=1.0,
		fake=True,
	)
	base.update(overrides)
	return HuntConfig(**base)


def test_grid_covers_corners():
	ws = WorkspaceMm(0, 1000, 0, 1000)
	pts = grid_points(ws, 500)
	assert (0.0, 0.0) in pts
	assert (1000.0, 1000.0) in pts


def test_clip_xy():
	ws = WorkspaceMm(0, 100, 0, 100)
	assert clip_xy(-10, 50, ws) == (0.0, 50.0)
	assert clip_xy(110, 110, ws) == (100.0, 100.0)


def test_plan_has_grid_and_corner_flees():
	plan = build_coverage_plan(_cfg())
	kinds = {s.kind for s in plan}
	assert "grid" in kinds
	assert "flee" in kinds
	assert "sweep" in kinds
	# Four corners × flees
	flee_labels = [s.label for s in plan if s.kind == "flee"]
	assert any("corner_0" in lab for lab in flee_labels)
	assert any("corner_3" in lab for lab in flee_labels)


def test_corner_points_inset():
	ws = WorkspaceMm(0, 1000, 0, 1000)
	corners = corner_points(ws, 50)
	assert corners[0] == (50.0, 50.0)
	assert corners[2] == (950.0, 950.0)


def test_extend_plan_pads_min_duration():
	cfg = _cfg(min_run_s=10_000.0, coverage_grid_mm=500.0)
	plan = build_coverage_plan(cfg)
	start = PoseMm(500, 500)
	est = estimate_plan_duration_s(plan, start)
	extended = extend_plan_for_min_duration(plan, cfg, est)
	assert len(extended) > len(plan)


def test_fake_gantry_move_and_full_plan_builds():
	cfg = _cfg(min_run_s=1.0, coverage_grid_mm=500.0)
	g = FakeGantry(cfg.workspace)
	g.connect()
	g.home()
	dt = g.move_abs_mm(100, 100, 0.5)
	assert dt >= 0
	pose = g.get_xy()
	assert abs(pose.x - 100) < 1e-6
	plan = build_full_plan(cfg, pose)
	assert len(plan) > 0
	g.close()
