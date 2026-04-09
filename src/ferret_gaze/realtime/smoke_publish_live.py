"""
Step 8 smoke publisher for Unreal realtime validation.

Publishes synthetic gaze packets over ZMQ PUB as msgpack:
- multipart frames: [topic, msgpack_payload]
- payload keys match RealtimeGazePacket / Unreal worker parser fields
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
	# Allow running as a direct script path by adding repo root.
	repo_root = Path(__file__).resolve().parents[3]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _build_payload(seq: int, now_ns: int) -> dict[str, object]:
	"""Build one synthetic live packet with smooth oscillatory motion."""
	yaw = 0.35 * math.sin(seq * 0.04)
	skull_quat_wxyz = (
		math.cos(yaw / 2.0),
		0.0,
		0.0,
		math.sin(yaw / 2.0),
	)
	left_origin = (-1.2, 3.5, 152.0)
	right_origin = (1.2, 3.5, 152.0)
	left_dir = (math.cos(yaw), math.sin(yaw), 0.05)
	right_dir = (math.cos(yaw), math.sin(yaw), 0.05)
	return {
		"seq": seq,
		"capture_utc_ns": now_ns,
		"process_start_ns": now_ns,
		"publish_utc_ns": now_ns,
		"skull_position_xyz": (0.0, 0.0, 150.0),
		"skull_quaternion_wxyz": skull_quat_wxyz,
		"left_eye_origin_xyz": left_origin,
		"left_gaze_direction_xyz": left_dir,
		"right_eye_origin_xyz": right_origin,
		"right_gaze_direction_xyz": right_dir,
		"confidence": 0.95,
	}


def run_smoke_publisher(endpoint: str, topic: str, hz: float, seconds: float) -> None:
	"""Publish synthetic msgpack packets for a fixed duration."""
	try:
		import msgpack  # type: ignore[import-not-found]
		import zmq  # type: ignore[import-not-found]
	except ImportError as exc:
		raise ImportError(
			"Smoke publisher requires msgpack and pyzmq. Install with: uv add msgpack pyzmq"
		) from exc

	context = zmq.Context.instance()
	socket = context.socket(zmq.PUB)
	socket.bind(endpoint)
	topic_bytes = topic.encode("utf-8")
	period_s = 1.0 / hz
	packet_count = max(1, int(seconds * hz))
	logger.info(
		"Starting smoke publisher endpoint=%s topic=%s hz=%.2f seconds=%.2f packets=%d",
		endpoint,
		topic,
		hz,
		seconds,
		packet_count,
	)
	try:
		for seq in range(packet_count):
			start_ns = time.time_ns()
			payload = _build_payload(seq=seq, now_ns=start_ns)
			packed = msgpack.packb(payload, use_bin_type=True)
			socket.send_multipart([topic_bytes, packed])
			elapsed_s = (time.time_ns() - start_ns) / 1e9
			sleep_s = max(0.0, period_s - elapsed_s)
			if sleep_s > 0:
				time.sleep(sleep_s)
	finally:
		socket.close(linger=0)
		logger.info("Smoke publisher finished")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Step 8 Unreal realtime smoke publisher")
	parser.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
	parser.add_argument("--topic", default="gaze.live")
	parser.add_argument("--hz", type=float, default=60.0)
	parser.add_argument("--seconds", type=float, default=120.0)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_smoke_publisher(
		endpoint=args.endpoint,
		topic=args.topic,
		hz=args.hz,
		seconds=args.seconds,
	)
