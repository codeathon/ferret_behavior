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
	session_root: Path,
	geometry_path: Path,
	*,
	calibration_toml_path: Path | None = None,
) -> dict:
	"""Assemble manifest dict for FerretArenaRender plugin."""
	inputs = resolve_arena_session_inputs(session_root, calibration_toml_path=calibration_toml_path)
	geometry = ArenaRenderGeometry.from_json_file(geometry_path)
	arena_render_dir = inputs.full_recording / "arena_render"
	timeline_path = arena_render_dir / "stimulus_timeline.json"
	texture_root = arena_render_dir / "wall_textures"
	if not timeline_path.is_file():
		raise FileNotFoundError(f"Missing timeline: {timeline_path}")
	if not texture_root.is_dir():
		raise FileNotFoundError(f"Missing textures: {texture_root}")

	timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
	arena_mm = geometry.arena_mm
	return {
		"version": 1,
		"session_root": str(inputs.session_root),
		"texture_root": str(texture_root.resolve()),
		"timeline_json": str(timeline_path.resolve()),
		"frame_count": timeline.get("frame_count", 0),
		"walls": timeline.get("walls", [wall.id for wall in geometry.walls]),
		"playback_fps": _estimate_fps(inputs.sync_video_dir),
		"units": "cm",
		"arena": {
			"center_cm": [v * MM_TO_CM for v in arena_mm.center],
			"half_size_cm": [v * MM_TO_CM for v in arena_mm.half_size],
		},
		"wall_screens": _wall_export_cm(geometry),
		"cameras": _export_cameras_cm(inputs.calibration_toml),
	}


def write_unreal_arena_manifest(manifest: dict, output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(manifest, indent="\t") + "\n", encoding="utf-8")
	logger.info("Wrote Unreal manifest: %s", output_path)
	return output_path


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Export Unreal arena manifest from session outputs")
	parser.add_argument("--session", type=Path, required=True, help="Session root path")
	parser.add_argument(
		"--geometry",
		type=Path,
		default=None,
		help="arena_geometry.json (default: session/arena_geometry.json)",
	)
	parser.add_argument("--calibration-toml", type=Path, default=None)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output path (default: full_recording/arena_render/unreal_arena_manifest.json)",
	)
	args = parser.parse_args(argv)

	session_root = args.session.resolve()
	geometry_path = args.geometry or (session_root / "arena_geometry.json")
	if not geometry_path.is_file():
		print(f"Geometry not found: {geometry_path}", file=sys.stderr)
		return 1

	try:
		manifest = build_unreal_arena_manifest(
			session_root,
			geometry_path,
			calibration_toml_path=args.calibration_toml,
		)
	except (FileNotFoundError, RuntimeError) as exc:
		print(exc, file=sys.stderr)
		return 1

	output = args.output or (session_root / "full_recording" / "arena_render" / "unreal_arena_manifest.json")
	write_unreal_arena_manifest(manifest, output.resolve())
	return 0


if __name__ == "__main__":
	sys.exit(main())
