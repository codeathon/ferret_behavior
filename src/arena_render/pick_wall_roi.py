"""
Interactive wall ROI picker for arena_geometry.json.

Usage:
    uv run python -m src.arena_render.pick_wall_roi --video /path/to/24676894_....mp4
    uv run python -m src.arena_render.pick_wall_roi --session $SESSION_ROOT --camera 24676894

Requires a display (run on lab desktop or ssh -X). Saves a crop preview to /tmp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.arena_render.session_inputs import discover_sync_video_dir, find_video_by_camera_name, list_overhead_videos
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
	"""Load one video frame; raise with a clear message if OpenCV cannot decode it."""
	if not video_path.is_file():
		raise FileNotFoundError(f"Video not found: {video_path}")

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(
			f"OpenCV failed to open video: {video_path}\n"
			"Check the path and that opencv/ffmpeg can read this codec."
		)

	if frame_index > 0:
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

	ok, frame = cap.read()
	cap.release()

	if not ok or frame is None or frame.size == 0:
		raise RuntimeError(
			f"Could not read frame {frame_index} from {video_path.name}.\n"
			"Try --frame-index 0 or a different mp4 from synchronized_corrected_videos/."
		)
	return frame


def _resolve_video_path(session_root: Path, camera_name: str) -> Path:
	sync_dir = discover_sync_video_dir(session_root)
	overhead = list_overhead_videos(sync_dir)
	match = find_video_by_camera_name(overhead, camera_name)
	return match.path


def _pick_roi(frame: np.ndarray, window_title: str) -> tuple[int, int, int, int]:
	"""Open selectROI GUI; returns (x, y, w, h)."""
	if frame.shape[0] == 0 or frame.shape[1] == 0:
		raise RuntimeError("Frame has zero width or height — cannot show ROI picker.")

	# selectROI calls imshow internally; needs DISPLAY when running over SSH.
	roi = cv2.selectROI(window_title, frame, fromCenter=False, showCrosshair=True)
	cv2.destroyAllWindows()
	x, y, w, h = (int(v) for v in roi)
	if w <= 0 or h <= 0:
		raise RuntimeError("No ROI selected (zero width/height). Drag a box and press Enter.")
	return x, y, w, h


def _save_preview(frame: np.ndarray, x: int, y: int, w: int, h: int, out_path: Path) -> None:
	crop = frame[y : y + h, x : x + w]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_path), crop)
	logger.info("Wrote crop preview: %s", out_path)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Pick wall ROI from a synced overhead video frame")
	parser.add_argument("--video", type=Path, default=None, help="Path to synchronized overhead mp4")
	parser.add_argument("--session", type=Path, default=None, help="Session root (use with --camera)")
	parser.add_argument(
		"--camera",
		type=str,
		default=None,
		help="Basler serial substring, e.g. 24676894 (use with --session)",
	)
	parser.add_argument("--frame-index", type=int, default=0, help="Frame to display (default: 0)")
	parser.add_argument("--wall-id", type=str, default="north", help="Label for JSON snippet output")
	parser.add_argument(
		"--preview-out",
		type=Path,
		default=Path("/tmp/wall_roi_preview.jpg"),
		help="Where to save cropped preview image",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)

	if args.video is not None:
		video_path = args.video.resolve()
	elif args.session is not None and args.camera is not None:
		video_path = _resolve_video_path(args.session.resolve(), args.camera)
	else:
		print("Provide --video PATH or both --session and --camera", file=sys.stderr)
		return 1

	try:
		frame = _read_frame(video_path, args.frame_index)
	except (FileNotFoundError, RuntimeError) as exc:
		print(exc, file=sys.stderr)
		return 1

	print(f"Video: {video_path}")
	print(f"Frame: {args.frame_index}  size: {frame.shape[1]}x{frame.shape[0]}")
	print("Drag a box around the wall screen, then press Enter or Space.")

	try:
		x, y, w, h = _pick_roi(frame, f"{args.wall_id} wall — drag box, Enter to confirm")
	except RuntimeError as exc:
		print(exc, file=sys.stderr)
		return 1

	_save_preview(frame, x, y, w, h, args.preview_out)

	roi_json = {"x": x, "y": y, "w": w, "h": h}
	snippet = {
		"id": args.wall_id,
		"source_camera_name": args.camera or video_path.stem.split("_")[0],
		"roi_px": roi_json,
	}
	print("\nPaste into arena_geometry.json walls[]:\n")
	print(json.dumps(snippet, indent="\t"))

	return 0


if __name__ == "__main__":
	sys.exit(main())
