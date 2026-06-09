"""
Build an Unreal-friendly manifest from arena_render outputs.

Packages paths, wall placement (cm), and camera poses so the FerretArenaRender
plugin can load stimulus_timeline.json without hard-coded lab paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import toml

from src.arena_render.arena_config import ArenaRenderGeometry
from src.arena_render.session_inputs import resolve_arena_session_inputs
from src.arena_render.timeline_exporter import merge_pose_into_timeline, write_stimulus_timeline
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

MM_TO_CM = 0.1


def _estimate_fps(sync_video_dir: Path) -> float:
	"""Estimate playback fps from first overhead mp4 timestamp spacing."""
	timestamp_files = sorted(sync_video_dir.glob("*_timestamps_utc.npy"))
	if not timestamp_files:
		return 90.0
	timestamps = np.load(timestamp_files[0], allow_pickle=True)
	if len(timestamps) < 2:
		return 90.0
	delta_ns = int(timestamps[1]) - int(timestamps[0])
	if delta_ns <= 0:
		return 90.0
	return float(1e9 / delta_ns)


def _estimate_fps_from_timeline(timeline: dict) -> float | None:
	"""Fallback fps when sync videos are unavailable (e.g. Mac-only texture export)."""
	frames = timeline.get("frames", [])
	if len(frames) < 2:
		return None
	t0 = frames[0].get("timestamp_utc_ns")
	t1 = frames[1].get("timestamp_utc_ns")
	if t0 is None or t1 is None:
		return None
	delta_ns = int(t1) - int(t0)
	if delta_ns <= 0:
		return None
	return float(1e9 / delta_ns)


def _export_cameras_cm(calibration_toml: Path) -> list[dict]:
	"""Export camera poses for Unreal (positions in cm, basis vectors)."""
	raw = toml.load(calibration_toml)
	cameras: list[dict] = []
	for key in sorted(k for k in raw if isinstance(k, str) and k.startswith("cam_")):
		block = raw[key]
		pos_mm = block["world_position"]
		orient = block["world_orientation"]
		cameras.append(
			{
				"id": key,
				"name": str(block.get("name", key)),
				"position_cm": [float(v) * MM_TO_CM for v in pos_mm],
				"world_orientation": orient,
			}
		)
	return cameras


def _wall_export_cm(geometry: ArenaRenderGeometry) -> list[dict]:
	"""Wall screen placement converted mm -> cm for Unreal."""
	walls: list[dict] = []
	for wall in geometry.walls:
		walls.append(
			{
				"id": wall.id,
				"screen_center_cm": [v * MM_TO_CM for v in wall.screen_center_mm],
				"screen_half_size_cm": [v * MM_TO_CM for v in wall.screen_half_size_mm],
				"normal": wall.normal,
				"texture_width_px": wall.texture_width_px,
				"texture_height_px": wall.texture_height_px,
			}
		)
	return walls


def build_unreal_arena_manifest(
	session_root: Path | None,
	geometry_path: Path,
	*,
	arena_render_dir: Path | None = None,
	calibration_toml_path: Path | None = None,
	playback_fps: float | None = None,
) -> dict:
	"""Assemble manifest dict for FerretArenaRender plugin."""
	geometry = ArenaRenderGeometry.from_json_file(geometry_path)
	cameras: list[dict] = []
	resolved_fps = playback_fps

	if arena_render_dir is not None:
		# Mac/local export: textures + timeline copied without full session tree.
		arena_render_dir = arena_render_dir.resolve()
		timeline_path = arena_render_dir / "stimulus_timeline.json"
		texture_root = arena_render_dir / "wall_textures"
		manifest_session_root = session_root.resolve() if session_root is not None else arena_render_dir
		if calibration_toml_path is not None and calibration_toml_path.is_file():
			cameras = _export_cameras_cm(calibration_toml_path)
		else:
			logger.warning(
				"No calibration TOML on this machine; manifest cameras[] will be empty. "
				"Pass --calibration-toml after rsyncing the file from the lab."
			)
	else:
		if session_root is None:
			raise ValueError("Either session_root or arena_render_dir is required")
		inputs = resolve_arena_session_inputs(session_root, calibration_toml_path=calibration_toml_path)
		arena_render_dir = inputs.full_recording / "arena_render"
		timeline_path = arena_render_dir / "stimulus_timeline.json"
		texture_root = arena_render_dir / "wall_textures"
		manifest_session_root = inputs.session_root
		if inputs.calibration_toml.is_file():
			cameras = _export_cameras_cm(inputs.calibration_toml)
		if resolved_fps is None:
			resolved_fps = _estimate_fps(inputs.sync_video_dir)

	if not timeline_path.is_file():
		raise FileNotFoundError(f"Missing timeline: {timeline_path}")
	if not texture_root.is_dir():
		raise FileNotFoundError(f"Missing textures: {texture_root}")

	timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
	if resolved_fps is None:
		resolved_fps = _estimate_fps_from_timeline(timeline) or 90.0

	arena_mm = geometry.arena_mm
	manifest_version = 2 if timeline.get("has_pose") else 1
	return {
		"version": manifest_version,
		"session_root": str(manifest_session_root),
		"texture_root": str(texture_root.resolve()),
		"timeline_json": str(timeline_path.resolve()),
		"frame_count": timeline.get("frame_count", 0),
		"walls": timeline.get("walls", [wall.id for wall in geometry.walls]),
		"playback_fps": resolved_fps,
		"units": "cm",
		"arena": {
			"center_cm": [v * MM_TO_CM for v in arena_mm.center],
			"half_size_cm": [v * MM_TO_CM for v in arena_mm.half_size],
		},
		"wall_screens": _wall_export_cm(geometry),
		"cameras": cameras,
		"has_pose": bool(timeline.get("has_pose")),
		"pose_units": timeline.get("pose_units", "cm"),
	}


def write_unreal_arena_manifest(manifest: dict, output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(manifest, indent="\t") + "\n", encoding="utf-8")
	logger.info("Wrote Unreal manifest: %s", output_path)
	return output_path


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Export Unreal arena manifest from session outputs")
	parser.add_argument("--session", type=Path, default=None, help="Lab session root path")
	parser.add_argument(
		"--arena-render-dir",
		type=Path,
		default=None,
		help="Local arena_render folder (textures + stimulus_timeline.json)",
	)
	parser.add_argument(
		"--geometry",
		type=Path,
		default=None,
		help="arena_geometry.json (default: session/arena_geometry.json)",
	)
	parser.add_argument("--calibration-toml", type=Path, default=None)
	parser.add_argument("--playback-fps", type=float, default=None)
	parser.add_argument(
		"--analyzable-output",
		type=Path,
		default=None,
		help="Merge skull+gaze from analyzable_output/ before writing manifest",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output path (default: arena_render/unreal_arena_manifest.json)",
	)
	args = parser.parse_args(argv)

	if args.session is None and args.arena_render_dir is None:
		print("Provide --session (lab) or --arena-render-dir (local Mac copy)", file=sys.stderr)
		return 1

	session_root = args.session.resolve() if args.session is not None else None
	arena_render_dir = args.arena_render_dir.resolve() if args.arena_render_dir is not None else None
	geometry_path = args.geometry or (
		(session_root / "arena_geometry.json") if session_root is not None else None
	)
	if geometry_path is None or not geometry_path.is_file():
		print(f"Geometry not found: {geometry_path}", file=sys.stderr)
		return 1

	try:
		if args.analyzable_output is not None:
			if arena_render_dir is None:
				arena_render_dir = (
					session_root / "full_recording" / "arena_render"
					if session_root is not None
					else None
				)
			if arena_render_dir is None:
				print("--arena-render-dir or --session required with --analyzable-output", file=sys.stderr)
				return 1
			timeline_path = arena_render_dir / "stimulus_timeline.json"
			if not timeline_path.is_file():
				print(f"Missing timeline: {timeline_path}", file=sys.stderr)
				return 1
			timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
			timeline = merge_pose_into_timeline(timeline, args.analyzable_output.resolve())
			write_stimulus_timeline(timeline, timeline_path)
			logger.info("Pose merged into %s", timeline_path)

		manifest = build_unreal_arena_manifest(
			session_root,
			geometry_path,
			arena_render_dir=arena_render_dir,
			calibration_toml_path=args.calibration_toml,
			playback_fps=args.playback_fps,
		)
	except (FileNotFoundError, RuntimeError, ValueError) as exc:
		print(exc, file=sys.stderr)
		return 1

	if args.output is not None:
		output = args.output
	elif arena_render_dir is not None:
		output = arena_render_dir / "unreal_arena_manifest.json"
	else:
		output = session_root / "full_recording" / "arena_render" / "unreal_arena_manifest.json"
	write_unreal_arena_manifest(manifest, output.resolve())
	return 0


if __name__ == "__main__":
	sys.exit(main())
