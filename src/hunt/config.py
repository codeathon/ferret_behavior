"""Load hunt / gantry JSON config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceMm:
	x_min: float
	x_max: float
	y_min: float
	y_max: float


@dataclass(frozen=True)
class HuntConfig:
	"""Soft-limited arena + motion params for open-loop prey scripts."""

	port: str
	baud_rate: int
	device_address: int
	axis_x: int
	axis_y: int
	# Optional second X for XXY lockstep; 0 = unused.
	axis_x2: int
	lockstep_group: int
	workspace: WorkspaceMm
	coverage_grid_mm: float
	min_run_s: float
	speeds_mps: tuple[float, ...]
	distances_mm: tuple[float, ...]
	bearings_deg: tuple[float, ...]
	corner_margin_mm: float
	flee_distances_mm: tuple[float, ...]
	hunt_rearm_s: float
	max_accel_mps2: float
	settle_tol_mm: float
	fake: bool


def load_hunt_config(path: str | Path) -> HuntConfig:
	raw = json.loads(Path(path).read_text())
	ws = raw["workspace"]
	return HuntConfig(
		port=str(raw.get("port", "/dev/ttyUSB0")),
		baud_rate=int(raw.get("baud_rate", 115200)),
		device_address=int(raw.get("device_address", 1)),
		axis_x=int(raw.get("axis_x", 1)),
		axis_y=int(raw.get("axis_y", 2)),
		axis_x2=int(raw.get("axis_x2", 0)),
		lockstep_group=int(raw.get("lockstep_group", 1)),
		workspace=WorkspaceMm(
			x_min=float(ws["x_min"]),
			x_max=float(ws["x_max"]),
			y_min=float(ws["y_min"]),
			y_max=float(ws["y_max"]),
		),
		coverage_grid_mm=float(raw.get("coverage_grid_mm", 100.0)),
		min_run_s=float(raw.get("min_run_s", 300.0)),
		speeds_mps=tuple(float(v) for v in raw.get("speeds_mps", [0.1, 0.3, 0.5])),
		distances_mm=tuple(float(v) for v in raw.get("distances_mm", [50, 200, 400])),
		bearings_deg=tuple(
			float(v) for v in raw.get("bearings_deg", [0, 45, 90, 135, 180, 225, 270, 315])
		),
		corner_margin_mm=float(raw.get("corner_margin_mm", 50.0)),
		flee_distances_mm=tuple(
			float(v) for v in raw.get("flee_distances_mm", [200, 400, 600])
		),
		hunt_rearm_s=float(raw.get("hunt_rearm_s", 1.5)),
		max_accel_mps2=float(raw.get("max_accel_mps2", 2.0)),
		settle_tol_mm=float(raw.get("settle_tol_mm", 1.0)),
		fake=bool(raw.get("fake", False)),
	)
