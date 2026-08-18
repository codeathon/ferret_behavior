"""Execute coverage segments and append timing CSV rows."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.hunt.config import HuntConfig
from src.hunt.coverage_planner import Segment, build_coverage_plan, extend_plan_for_min_duration
from src.hunt.zaber_client import Gantry, PoseMm


CSV_FIELDS = [
	"t_host",
	"kind",
	"label",
	"cmd_x",
	"cmd_y",
	"speed_mps",
	"actual_x",
	"actual_y",
	"move_s",
	"err_mm",
	"pause_s",
]


def estimate_plan_duration_s(plan: list[Segment], start: PoseMm) -> float:
	"""Crude lower-bound travel time for min_run_s padding."""
	x, y = start.x, start.y
	total = 0.0
	for seg in plan:
		dist_m = ((seg.x1 - x) ** 2 + (seg.y1 - y) ** 2) ** 0.5 / 1000.0
		total += dist_m / max(seg.speed_mps, 1e-3) + seg.pause_s
		x, y = seg.x1, seg.y1
	return total


def build_full_plan(cfg: HuntConfig, start: PoseMm) -> list[Segment]:
	plan = build_coverage_plan(cfg)
	est = estimate_plan_duration_s(plan, start)
	return extend_plan_for_min_duration(plan, cfg, est)


def run_segments(
	gantry: Gantry | None,
	plan: list[Segment],
	cfg: HuntConfig,
	log_path: Path,
	*,
	dry_run: bool = False,
) -> int:
	"""Run plan; return number of segments executed. Enforces min_run_s wall clock."""
	log_path.parent.mkdir(parents=True, exist_ok=True)
	t_wall0 = time.perf_counter()
	n = 0
	with log_path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
		writer.writeheader()
		if dry_run:
			for seg in plan:
				writer.writerow(_row_dry(seg))
				n += 1
			return n
		if gantry is None:
			raise ValueError("gantry required unless dry_run")

		for seg in plan:
			n += 1
			row = _execute_one(gantry, seg)
			writer.writerow(row)
			f.flush()
			if seg.pause_s > 0:
				time.sleep(seg.pause_s)

		# If checklist finished early vs min_run_s, repeat grid until wall time met.
		while time.perf_counter() - t_wall0 < cfg.min_run_s:
			grid = [s for s in plan if s.kind == "grid"] or plan[: max(1, len(plan) // 4)]
			for seg in grid:
				if time.perf_counter() - t_wall0 >= cfg.min_run_s:
					break
				n += 1
				writer.writerow(_execute_one(gantry, seg))
				f.flush()
	return n


def _execute_one(gantry: Gantry, seg: Segment) -> dict:
	t0 = time.perf_counter()
	move_s = gantry.move_abs_mm(seg.x1, seg.y1, seg.speed_mps)
	pose = gantry.get_xy()
	err = ((pose.x - seg.x1) ** 2 + (pose.y - seg.y1) ** 2) ** 0.5
	return {
		"t_host": f"{t0:.6f}",
		"kind": seg.kind,
		"label": seg.label,
		"cmd_x": f"{seg.x1:.3f}",
		"cmd_y": f"{seg.y1:.3f}",
		"speed_mps": f"{seg.speed_mps:.4f}",
		"actual_x": f"{pose.x:.3f}",
		"actual_y": f"{pose.y:.3f}",
		"move_s": f"{move_s:.4f}",
		"err_mm": f"{err:.3f}",
		"pause_s": f"{seg.pause_s:.3f}",
	}


def _row_dry(seg: Segment) -> dict:
	return {
		"t_host": "0",
		"kind": seg.kind,
		"label": seg.label,
		"cmd_x": f"{seg.x1:.3f}",
		"cmd_y": f"{seg.y1:.3f}",
		"speed_mps": f"{seg.speed_mps:.4f}",
		"actual_x": "",
		"actual_y": "",
		"move_s": "",
		"err_mm": "",
		"pause_s": f"{seg.pause_s:.3f}",
	}
