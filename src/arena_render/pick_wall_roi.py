"""
Interactive wall ROI picker for arena_geometry.json.

OpenCV's Qt GUI often breaks on Linux (thread/font errors). Default backend is
matplotlib, which is more reliable on the lab desktop and over ssh -X.

Usage:
    uv run python -m src.arena_render.pick_wall_roi --session $SESSION_ROOT --camera 24676894
    uv run python -m src.arena_render.pick_wall_roi --mode export --session $SESSION_ROOT --camera 24676894
    uv run python -m src.arena_render.pick_wall_roi --roi 412,180,320,240 --video /path/to.mp4
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


def _parse_roi_string(roi_text: str) -> tuple[int, int, int, int]:
	"""Parse ``x,y,w,h`` from CLI."""
	parts = [part.strip() for part in roi_text.split(",")]
	if len(parts) != 4:
		raise ValueError("ROI must be four comma-separated integers: x,y,w,h")
	x, y, w, h = (int(part) for part in parts)
	if w <= 0 or h <= 0:
		raise ValueError("ROI width and height must be positive")
	return x, y, w, h


def _extents_to_roi(extents: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
	"""Convert RectangleSelector extents (xmin, xmax, ymin, ymax) to x,y,w,h."""
	xmin, xmax, ymin, ymax = extents
	x = int(round(xmin))
	y = int(round(ymin))
	w = int(round(xmax - xmin))
	h = int(round(ymax - ymin))
	if w <= 0 or h <= 0:
		raise RuntimeError("No ROI selected. Drag a rectangle, then close the window.")
	return x, y, w, h


def _pick_roi_matplotlib(frame: np.ndarray, window_title: str) -> tuple[int, int, int, int]:
	"""Drag a box with matplotlib (avoids OpenCV Qt highgui issues)."""
	import matplotlib.pyplot as plt
	from matplotlib.widgets import RectangleSelector

	if frame.shape[0] == 0 or frame.shape[1] == 0:
		raise RuntimeError("Frame has zero width or height — cannot show ROI picker.")

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	fig, axis = plt.subplots(figsize=(14, 10))
	axis.imshow(rgb)
	axis.set_title(
		f"{window_title}\n"
		"Click-drag a box around the wall screen, then close this window"
	)
	# matplotlib >=3.7 removed drawtype; read extents after plt.show() for compatibility.
	selector = RectangleSelector(
		axis,
		onselect=lambda *args, **kwargs: None,
		useblit=True,
		button=[1],
		minspanx=5,
		minspany=5,
		spancoords="pixels",
		interactive=True,
	)
	plt.show()
	return _extents_to_roi(selector.extents)


def _pick_roi_opencv(frame: np.ndarray, window_title: str) -> tuple[int, int, int, int]:
	"""OpenCV selectROI — may fail when Qt backend is broken on Linux."""
	if frame.shape[0] == 0 or frame.shape[1] == 0:
		raise RuntimeError("Frame has zero width or height — cannot show ROI picker.")
	roi = cv2.selectROI(window_title, frame, fromCenter=False, showCrosshair=True)
	cv2.destroyAllWindows()
	x, y, w, h = (int(v) for v in roi)
	if w <= 0 or h <= 0:
		raise RuntimeError("No ROI selected (zero width/height). Drag a box and press Enter.")
	return x, y, w, h


def pick_roi(frame: np.ndarray, window_title: str, backend: str) -> tuple[int, int, int, int]:
	"""Dispatch to matplotlib (default) or opencv ROI UI."""
	if backend == "matplotlib":
		return _pick_roi_matplotlib(frame, window_title)
	if backend == "opencv":
		return _pick_roi_opencv(frame, window_title)
	raise ValueError(f"Unknown backend: {backend}")


def _save_preview(frame: np.ndarray, x: int, y: int, w: int, h: int, out_path: Path) -> None:
	crop = frame[y : y + h, x : x + w]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(out_path), crop)
	logger.info("Wrote crop preview: %s", out_path)


def _export_frame(frame: np.ndarray, frame_out: Path) -> None:
	frame_out.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(frame_out), frame)
	print(f"Saved frame: {frame_out}")
	print("Open in an image viewer, note x,y,w,h of the wall screen, then run:")
	print(f"  --roi X,Y,W,H --video ...  (or re-run with --roi after inspecting)")


def _print_snippet(
	wall_id: str,
	camera_name: str,
	x: int,
	y: int,
	w: int,
	h: int,
) -> None:
	snippet = {
		"id": wall_id,
		"source_camera_name": camera_name,
		"roi_px": {"x": x, "y": y, "w": w, "h": h},
	}
	print("\nPaste into arena_geometry.json walls[]:\n")
	print(json.dumps(snippet, indent="\t"))


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Pick wall ROI from a synced overhead video frame")
	parser.add_argument("--video", type=Path, default=None, help="Path to synchronized overhead mp4")
	parser.add_argument("--session", type=Path, default=None, help="Session root (use with --camera)")
	parser.add_argument("--camera", type=str, default=None, help="Basler serial substring, e.g. 24676894")
	parser.add_argument("--frame-index", type=int, default=0, help="Frame to display (default: 0)")
	parser.add_argument("--wall-id", type=str, default="north", help="Label for JSON snippet output")
	parser.add_argument(
		"--backend",
		choices=("matplotlib", "opencv"),
		default="matplotlib",
		help="ROI UI backend (default: matplotlib; opencv Qt often breaks on Linux)",
	)
	parser.add_argument(
		"--mode",
		choices=("pick", "export"),
		default="pick",
		help="pick=interactive ROI; export=save frame jpg only (no GUI)",
	)
	parser.add_argument("--roi", type=str, default=None, help="Skip picker; use x,y,w,h and write preview")
	parser.add_argument(
		"--frame-out",
		type=Path,
		default=Path("/tmp/wall_roi_frame.jpg"),
		help="Output path for --mode export",
	)
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

	camera_name = args.camera or video_path.stem.split("_")[0]

	try:
		frame = _read_frame(video_path, args.frame_index)
	except (FileNotFoundError, RuntimeError) as exc:
		print(exc, file=sys.stderr)
		return 1

	print(f"Video: {video_path}")
	print(f"Frame: {args.frame_index}  size: {frame.shape[1]}x{frame.shape[0]}")

	if args.mode == "export":
		_export_frame(frame, args.frame_out)
		return 0

	if args.roi is not None:
		try:
			x, y, w, h = _parse_roi_string(args.roi)
		except ValueError as exc:
			print(exc, file=sys.stderr)
			return 1
	else:
		print(f"Using {args.backend} picker — drag a box, then close the window.")
		try:
			x, y, w, h = pick_roi(frame, f"{args.wall_id} wall", args.backend)
		except RuntimeError as exc:
			print(exc, file=sys.stderr)
			print("\nFallback: save frame without GUI, measure ROI in an image editor:")
			print(f"  --mode export --frame-out {args.frame_out}")
			print("  then re-run with --roi x,y,w,h")
			return 1

	_save_preview(frame, x, y, w, h, args.preview_out)
	_print_snippet(args.wall_id, camera_name, x, y, w, h)
	return 0


if __name__ == "__main__":
	sys.exit(main())
