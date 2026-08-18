#!/usr/bin/env python3
"""Repeated small moves to measure Zaber command / settle timing."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from src.hunt.config import load_hunt_config
from src.hunt.timing_bench import run_latency_bench
from src.hunt.zaber_client import open_gantry


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/hunt_experiment.json")
	parser.add_argument("--repeats", type=int, default=20)
	parser.add_argument("--step-mm", type=float, default=20.0)
	parser.add_argument("--log", default=None)
	parser.add_argument("--fake", action="store_true")
	parser.add_argument("--no-home", action="store_true")
	args = parser.parse_args()

	cfg = load_hunt_config(args.config)
	if args.fake:
		cfg = replace(cfg, fake=True)

	log_path = Path(args.log) if args.log else _default_log("timing_bench")
	gantry = open_gantry(cfg)
	try:
		if not args.no_home:
			gantry.home()
		stats = run_latency_bench(
			gantry, cfg, log_path, repeats=args.repeats, step_mm=args.step_mm
		)
		print(f"n={stats['n']:.0f} p50_s={stats['p50_s']:.4f} p95_s={stats['p95_s']:.4f}")
		print(f"log={log_path}")
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
