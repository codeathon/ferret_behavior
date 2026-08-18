#!/usr/bin/env python3
"""Full-arena prey coverage + corner-flee stress script (Zaber gantry)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on path when launched as scripts/run_prey_move.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from src.hunt.config import load_hunt_config
from src.hunt.runner import build_full_plan, run_segments
from src.hunt.zaber_client import PoseMm, open_gantry


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config",
		default="configs/hunt_experiment.json",
		help="Path to hunt_experiment.json",
	)
	parser.add_argument("--min-run-s", type=float, default=None, help="Override min_run_s")
	parser.add_argument(
		"--log",
		default=None,
		help="CSV output path (default: data/hunt_logs/prey_move_*.csv)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Write planned segments only; no motion",
	)
	parser.add_argument(
		"--fake",
		action="store_true",
		help="Force FakeGantry (no serial)",
	)
	parser.add_argument("--no-home", action="store_true", help="Skip home()")
	args = parser.parse_args()

	cfg = load_hunt_config(args.config)
	if args.min_run_s is not None or args.fake:
		# Rebuild frozen dataclass with overrides.
		from dataclasses import replace

		updates = {}
		if args.min_run_s is not None:
			updates["min_run_s"] = args.min_run_s
		if args.fake:
			updates["fake"] = True
		cfg = replace(cfg, **updates)

	log_path = Path(args.log) if args.log else _default_log("prey_move")
	print(f"config={args.config} fake={cfg.fake} min_run_s={cfg.min_run_s}")
	print(f"log={log_path}")

	if args.dry_run:
		start = PoseMm(
			0.5 * (cfg.workspace.x_min + cfg.workspace.x_max),
			0.5 * (cfg.workspace.y_min + cfg.workspace.y_max),
		)
		plan = build_full_plan(cfg, start)
		n = run_segments(None, plan, cfg, log_path, dry_run=True)
		print(f"dry-run segments={n}")
		return

	gantry = open_gantry(cfg)
	try:
		if not args.no_home:
			print("homing…")
			gantry.home()
		start = gantry.get_xy()
		plan = build_full_plan(cfg, start)
		print(f"plan_segments={len(plan)}")
		n = run_segments(gantry, plan, cfg, log_path, dry_run=False)
		print(f"done segments={n} log={log_path}")
	finally:
		gantry.stop()
		gantry.close()


def _default_log(prefix: str) -> Path:
	stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	path = Path("data/hunt_logs") / f"{prefix}_{stamp}.csv"
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


if __name__ == "__main__":
	main()
