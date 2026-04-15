"""
Opt-in Unreal acceptance harness for live realtime transport.

This script launches Unreal Editor, runs the smoke publisher, and parses Unreal
stdout for transport and render-health telemetry lines:
- ``GazeSubscriber ...``
- ``GazeRender stats | ...``

It is intended for local/nightly acceptance, not default unit CI.
"""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
	repo_root = Path(__file__).resolve().parents[3]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

from src.ferret_gaze.realtime.smoke_publish_live import run_smoke_publisher
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

_SUB_RE = re.compile(
	r"GazeSubscriber (?P<event>\w+)\s+\|\s+connected=(?P<connected>\d+)\s+"
	r"reconnect_attempts=(?P<reconnect>\d+)\s+receive_errors=(?P<recv>\d+)\s+"
	r"dropped=(?P<dropped>\d+)\s+last_seq=(?P<last_seq>-?\d+)"
)
_RENDER_RE = re.compile(
	r"GazeRender stats \| applied=(?P<applied>\d+)\s+stale_drop=(?P<stale>\d+)\s+"
	r"conf_drop=(?P<conf>\d+)\s+age_ms\[p50=(?P<p50>[0-9.]+)\s+p95=(?P<p95>[0-9.]+)\s+last=(?P<last>[0-9.]+)\]"
)


@dataclass
class UnrealAcceptanceMetrics:
	connected_events: int = 0
	max_last_seq: int = -1
	max_receive_errors: int = 0
	max_dropped: int = 0
	render_stats_events: int = 0
	max_applied: int = 0
	max_stale_drop: int = 0
	max_conf_drop: int = 0
	max_age_p95_ms: float = 0.0


@dataclass
class UnrealAcceptanceThresholds:
	min_connected_events: int = 1
	min_last_seq: int = 30
	min_render_stats_events: int = 1
	min_applied_packets: int = 30
	max_receive_errors: int = 0
	max_stale_drop: int = 200
	max_conf_drop: int = 200
	max_age_p95_ms: float = 1000.0


def parse_unreal_log_line(line: str, metrics: UnrealAcceptanceMetrics) -> None:
	"""Update metrics from one Unreal log line when it matches known telemetry."""
	m = _SUB_RE.search(line)
	if m:
		if m.group("event") == "connected":
			metrics.connected_events += 1
		metrics.max_last_seq = max(metrics.max_last_seq, int(m.group("last_seq")))
		metrics.max_receive_errors = max(metrics.max_receive_errors, int(m.group("recv")))
		metrics.max_dropped = max(metrics.max_dropped, int(m.group("dropped")))
		return
	m2 = _RENDER_RE.search(line)
	if not m2:
		return
	metrics.render_stats_events += 1
	metrics.max_applied = max(metrics.max_applied, int(m2.group("applied")))
	metrics.max_stale_drop = max(metrics.max_stale_drop, int(m2.group("stale")))
	metrics.max_conf_drop = max(metrics.max_conf_drop, int(m2.group("conf")))
	metrics.max_age_p95_ms = max(metrics.max_age_p95_ms, float(m2.group("p95")))


def evaluate_unreal_acceptance(
	metrics: UnrealAcceptanceMetrics,
	thresholds: UnrealAcceptanceThresholds,
) -> tuple[bool, list[str]]:
	"""Return acceptance pass/fail with human-readable failure reasons."""
	failures: list[str] = []
	if metrics.connected_events < thresholds.min_connected_events:
		failures.append(f"connected_events {metrics.connected_events} < {thresholds.min_connected_events}")
	if metrics.max_last_seq < thresholds.min_last_seq:
		failures.append(f"max_last_seq {metrics.max_last_seq} < {thresholds.min_last_seq}")
	if metrics.render_stats_events < thresholds.min_render_stats_events:
		failures.append(
			f"render_stats_events {metrics.render_stats_events} < {thresholds.min_render_stats_events}"
		)
	if metrics.max_applied < thresholds.min_applied_packets:
		failures.append(f"max_applied {metrics.max_applied} < {thresholds.min_applied_packets}")
	if metrics.max_receive_errors > thresholds.max_receive_errors:
		failures.append(f"receive_errors {metrics.max_receive_errors} > {thresholds.max_receive_errors}")
	if metrics.max_stale_drop > thresholds.max_stale_drop:
		failures.append(f"stale_drop {metrics.max_stale_drop} > {thresholds.max_stale_drop}")
	if metrics.max_conf_drop > thresholds.max_conf_drop:
		failures.append(f"conf_drop {metrics.max_conf_drop} > {thresholds.max_conf_drop}")
	if metrics.max_age_p95_ms > thresholds.max_age_p95_ms:
		failures.append(f"age_p95_ms {metrics.max_age_p95_ms:.2f} > {thresholds.max_age_p95_ms:.2f}")
	return (len(failures) == 0), failures


def _reader_thread(stream, out_q: queue.Queue[str], stop_event: threading.Event) -> None:
	while not stop_event.is_set():
		line = stream.readline()
		if not line:
			break
		out_q.put(line.rstrip("\n"))


def _build_unreal_command(args: argparse.Namespace) -> list[str]:
	cmd = [str(args.unreal_bin), str(args.uproject)]
	if args.map:
		cmd.append(args.map)
	cmd.extend(["-game", "-log", "-stdout", "-FullStdOutLogOutput"])
	cmd.extend(args.unreal_arg)
	return cmd


def run_unreal_acceptance(args: argparse.Namespace) -> int:
	if not args.unreal_bin.exists():
		raise FileNotFoundError(f"Unreal binary not found: {args.unreal_bin}")
	if not args.uproject.exists():
		raise FileNotFoundError(f"uproject not found: {args.uproject}")
	metrics = UnrealAcceptanceMetrics()
	thresholds = UnrealAcceptanceThresholds(
		min_connected_events=args.min_connected_events,
		min_last_seq=args.min_last_seq,
		min_render_stats_events=args.min_render_stats_events,
		min_applied_packets=args.min_applied_packets,
		max_receive_errors=args.max_receive_errors,
		max_stale_drop=args.max_stale_drop,
		max_conf_drop=args.max_conf_drop,
		max_age_p95_ms=args.max_age_p95_ms,
	)
	cmd = _build_unreal_command(args)
	logger.info("Launching Unreal: %s", " ".join(cmd))
	proc = subprocess.Popen(
		cmd,
		cwd=str(args.uproject.parent),
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
	)
	stop_event = threading.Event()
	log_q: queue.Queue[str] = queue.Queue()
	reader = threading.Thread(target=_reader_thread, args=(proc.stdout, log_q, stop_event), daemon=True)
	reader.start()
	time.sleep(args.startup_wait_s)
	pub_thread = threading.Thread(
		target=run_smoke_publisher,
		kwargs={
			"endpoint": args.endpoint,
			"topic": args.topic,
			"hz": args.hz,
			"seconds": args.publisher_seconds,
			"motion": args.motion,
			"circle_period_s": args.circle_period_s,
		},
		daemon=True,
	)
	pub_thread.start()
	deadline = time.time() + args.timeout_s
	try:
		while time.time() < deadline and proc.poll() is None:
			try:
				line = log_q.get(timeout=0.25)
			except queue.Empty:
				continue
			if args.echo_unreal_log:
				print(line)
			parse_unreal_log_line(line, metrics)
		passed, failures = evaluate_unreal_acceptance(metrics, thresholds)
		if not passed:
			for f in failures:
				logger.error("UNREAL ACCEPTANCE FAIL: %s", f)
			return 1
		logger.info(
			"UNREAL ACCEPTANCE PASS: connected=%d max_last_seq=%d applied=%d p95_age_ms=%.2f",
			metrics.connected_events,
			metrics.max_last_seq,
			metrics.max_applied,
			metrics.max_age_p95_ms,
		)
		return 0
	finally:
		stop_event.set()
		if proc.poll() is None:
			proc.terminate()
			try:
				proc.wait(timeout=5.0)
			except subprocess.TimeoutExpired:
				proc.kill()
		pub_thread.join(timeout=2.0)
		reader.join(timeout=2.0)


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run Unreal realtime end-to-end acceptance.")
	p.add_argument("--unreal-bin", type=Path, required=True, help="Path to UnrealEditor binary.")
	p.add_argument("--uproject", type=Path, required=True, help="Path to .uproject.")
	p.add_argument("--map", type=str, default="", help="Optional map path (e.g. /Game/Maps/TestMap).")
	p.add_argument("--unreal-arg", action="append", default=[], help="Extra Unreal arg (repeatable).")
	p.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5556")
	p.add_argument("--topic", type=str, default="gaze.live")
	p.add_argument("--hz", type=float, default=60.0)
	p.add_argument("--publisher-seconds", type=float, default=8.0)
	p.add_argument("--motion", choices=("circle", "sine"), default="circle")
	p.add_argument("--circle-period-s", type=float, default=4.0)
	p.add_argument("--startup-wait-s", type=float, default=5.0)
	p.add_argument("--timeout-s", type=float, default=30.0)
	p.add_argument("--echo-unreal-log", action="store_true", help="Echo Unreal stdout lines.")
	p.add_argument("--min-connected-events", type=int, default=1)
	p.add_argument("--min-last-seq", type=int, default=30)
	p.add_argument("--min-render-stats-events", type=int, default=1)
	p.add_argument("--min-applied-packets", type=int, default=30)
	p.add_argument("--max-receive-errors", type=int, default=0)
	p.add_argument("--max-stale-drop", type=int, default=200)
	p.add_argument("--max-conf-drop", type=int, default=200)
	p.add_argument("--max-age-p95-ms", type=float, default=1000.0)
	return p.parse_args()


if __name__ == "__main__":
	raise SystemExit(run_unreal_acceptance(_parse_args()))

