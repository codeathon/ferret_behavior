"""Build open-loop coverage + corner-flee segment lists (no hardware I/O)."""

from __future__ import annotations

from dataclasses import dataclass

from src.hunt.config import HuntConfig
from src.hunt.workspace import (
	bearing_delta_mm,
	clip_move,
	corner_points,
	grid_points,
)


@dataclass(frozen=True)
class Segment:
	"""One absolute move: go to (x1,y1) at speed_mps from prior pose."""

	kind: str
	x1: float
	y1: float
	speed_mps: float
	label: str
	# Pause after this segment (hunt re-arm between flees).
	pause_s: float = 0.0


def build_coverage_plan(cfg: HuntConfig) -> list[Segment]:
	"""Grid lawnmower + direction/distance sweeps + corner flees.

	Why a fixed checklist: run_prey_move must cover the gantry footprint and
	especially flee-from-corner cases before (and while) satisfying min_run_s.
	"""
	ws = cfg.workspace
	plan: list[Segment] = []
	speed0 = cfg.speeds_mps[0] if cfg.speeds_mps else 0.2

	# --- Grid coverage (lawnmower order) ---
	grid = grid_points(ws, cfg.coverage_grid_mm)
	# Sort into rows for shorter travel.
	grid_sorted = sorted(grid, key=lambda p: (round(p[1], 3), round(p[0], 3)))
	for i, (x, y) in enumerate(grid_sorted):
		plan.append(
			Segment("grid", x, y, speed0, f"grid_{i}", pause_s=0.0)
		)

	# --- Direction × distance × speed sweeps from workspace center ---
	cx = 0.5 * (ws.x_min + ws.x_max)
	cy = 0.5 * (ws.y_min + ws.y_max)
	plan.append(Segment("goto", cx, cy, speed0, "center", pause_s=0.0))
	for speed in cfg.speeds_mps:
		for dist in cfg.distances_mm:
			for bearing in cfg.bearings_deg:
				dx, dy = bearing_delta_mm(dist, bearing)
				end = clip_move(cx, cy, dx, dy, ws)
				if end is None:
					continue
				ex, ey = end
				plan.append(
					Segment(
						"sweep",
						ex,
						ey,
						speed,
						f"sweep_b{bearing:.0f}_d{dist:.0f}_v{speed:.2f}",
					)
				)
				# Return toward center so the next bearing starts cleanly.
				plan.append(Segment("goto", cx, cy, speed, "return_center"))

	# --- Corner flees: approach each corner, flee toward interior ---
	corners = corner_points(ws, cfg.corner_margin_mm)
	flee_speed = cfg.speeds_mps[-1] if cfg.speeds_mps else speed0
	# Exit bearings from each corner: toward center + along walls (relative).
	for ci, (cx_c, cy_c) in enumerate(corners):
		plan.append(
			Segment("goto", cx_c, cy_c, speed0, f"corner_{ci}_approach")
		)
		# Vector from corner toward center defines "away from corner".
		away_deg = _bearing_deg(cx - cx_c, cy - cy_c)
		exit_bearings = (
			away_deg,
			away_deg - 45.0,
			away_deg + 45.0,
			away_deg - 90.0,
			away_deg + 90.0,
		)
		for fi, bearing in enumerate(exit_bearings):
			for dist in cfg.flee_distances_mm:
				dx, dy = bearing_delta_mm(dist, bearing)
				end = clip_move(cx_c, cy_c, dx, dy, ws)
				if end is None:
					continue
				ex, ey = end
				plan.append(
					Segment(
						"flee",
						ex,
						ey,
						flee_speed,
						f"corner_{ci}_flee_{fi}_d{dist:.0f}",
						pause_s=cfg.hunt_rearm_s,
					)
				)
				# Re-seat at corner for the next flee burst.
				plan.append(
					Segment("goto", cx_c, cy_c, speed0, f"corner_{ci}_reseat")
				)

	return plan


def extend_plan_for_min_duration(
	plan: list[Segment],
	cfg: HuntConfig,
	estimated_s: float,
) -> list[Segment]:
	"""If checklist is shorter than min_run_s, loop grid passes until long enough.

	estimated_s should be a conservative lower bound of plan duration (caller).
	"""
	if estimated_s >= cfg.min_run_s or not plan:
		return plan
	extra: list[Segment] = []
	grid = [s for s in plan if s.kind == "grid"]
	if not grid:
		grid = plan
	# Repeat grid blocks until we clearly exceed min_run_s under a crude estimate.
	need = cfg.min_run_s - estimated_s
	# Assume ~2 s per segment as a floor filler (real runner uses clocks).
	n = max(1, int(need / max(2.0, 0.5 * len(grid))) + 1)
	for k in range(n):
		for s in grid:
			extra.append(
				Segment(
					s.kind,
					s.x1,
					s.y1,
					s.speed_mps,
					f"{s.label}_loop{k}",
					pause_s=s.pause_s,
				)
			)
	return plan + extra


def _bearing_deg(dx: float, dy: float) -> float:
	import math

	return math.degrees(math.atan2(dy, dx)) % 360.0
