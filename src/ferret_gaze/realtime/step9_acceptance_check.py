"""
Step 9 acceptance checklist for realtime transport scaffolding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
	# Allow running as a direct script path by adding repo root.
	repo_root = Path(__file__).resolve().parents[3]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

from src.ferret_gaze.realtime import (
	create_realtime_publisher,
	format_latency_summary,
	load_realtime_runtime_config,
	run_realtime_transport_scaffold,
)
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def run_acceptance_check(config_path: Path | None = None) -> int:
	"""Run a minimal acceptance gate for realtime transport behavior."""
	config = load_realtime_runtime_config(config_path)
	publisher = create_realtime_publisher(
		backend=config.transport_backend,
		endpoint=config.transport_endpoint,
		topic=config.transport_topic,
	)
	summary = run_realtime_transport_scaffold(
		publisher=publisher,
		n_packets=config.transport_packets,
		hz=config.transport_hz,
		stale_threshold_ms=config.stale_threshold_ms,
	)
	logger.info(format_latency_summary(summary))

	failures: list[str] = []
	if summary.packet_count < config.transport_packets:
		failures.append("packet_count lower than expected")
	if summary.end_to_end_p95_ms > config.acceptance_max_p95_ms:
		failures.append(
			f"p95 end-to-end {summary.end_to_end_p95_ms:.2f}ms exceeds {config.acceptance_max_p95_ms:.2f}ms"
		)
	if summary.dropped_count > config.acceptance_max_dropped_count:
		failures.append(
			f"dropped packet count {summary.dropped_count} exceeds {config.acceptance_max_dropped_count}"
		)
	if summary.queue_overflow_count > config.acceptance_max_queue_overflow_count:
		failures.append(
			f"queue overflow count {summary.queue_overflow_count} exceeds {config.acceptance_max_queue_overflow_count}"
		)
	if summary.stage_error_count > config.acceptance_max_stage_error_count:
		failures.append(
			f"stage error count {summary.stage_error_count} exceeds {config.acceptance_max_stage_error_count}"
		)
	if summary.publish_error_count > config.acceptance_max_publish_error_count:
		failures.append(
			f"publish error count {summary.publish_error_count} exceeds {config.acceptance_max_publish_error_count}"
		)
	if summary.stale_count > 0:
		failures.append(f"stale packet count nonzero: {summary.stale_count}")

	if failures:
		for item in failures:
			logger.error("ACCEPTANCE FAIL: %s", item)
		return 1

	logger.info("ACCEPTANCE PASS: realtime scaffold metrics within thresholds")
	return 0


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Step 9 realtime acceptance check")
	parser.add_argument(
		"--config",
		type=Path,
		default=None,
		help="Optional JSON config path for RealtimeRuntimeConfig",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	raise SystemExit(run_acceptance_check(config_path=args.config))
