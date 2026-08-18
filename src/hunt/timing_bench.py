"""Host/device timing benches for single Zaber moves (no camera)."""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from src.hunt.config import HuntConfig
from src.hunt.zaber_client import Gantry
from src.hunt.workspace import clip_xy


def run_latency_bench(
	gantry: Gantry,
	cfg: HuntConfig,
	log_path: Path,
	*,
	repeats: int = 20,
	step_mm: float = 20.0,
) -> dict[str, float]:
	"""Repeated small absolute steps; return p50/p95 move durations."""
	log_path.parent.mkdir(parents=True, exist_ok=True)
	pose = gantry.get_xy()
	cx = 0.5 * (cfg.workspace.x_min + cfg.workspace.x_max)
	cy = 0.5 * (cfg.workspace.y_min + cfg.workspace.y_max)
	speed = cfg.speeds_mps[0] if cfg.speeds_mps else 0.2
	durs: list[float] = []

	with log_path.open("w", newline="") as f:
		w = csv.DictWriter(
			f,
			fieldnames=["i", "cmd_issue_s", "move_s", "x", "y"],
		)
		w.writeheader()
		# Start near center.
		gantry.move_abs_mm(cx, cy, speed)
		for i in range(repeats):
			target_x = cx + (step_mm if i % 2 == 0 else -step_mm)
			tx, ty = clip_xy(target_x, cy, cfg.workspace)
			t_issue = time.perf_counter()
			move_s = gantry.move_abs_mm(tx, ty, speed)
			pose = gantry.get_xy()
			durs.append(move_s)
			w.writerow(
				{
					"i": i,
					"cmd_issue_s": f"{t_issue:.6f}",
					"move_s": f"{move_s:.6f}",
					"x": f"{pose.x:.3f}",
					"y": f"{pose.y:.3f}",
				}
			)

	durs_sorted = sorted(durs)
	p50 = statistics.median(durs_sorted)
	p95 = durs_sorted[max(0, int(0.95 * (len(durs_sorted) - 1)))]
	return {"p50_s": p50, "p95_s": p95, "n": float(len(durs))}
