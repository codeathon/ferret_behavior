"""
CLI for offline arena reconstruction and wall-stimulus virtual rendering.

Usage:
    uv run python -m src.arena_render.run_arena_render --session /path/to/session validate
    uv run python -m src.arena_render.run_arena_render --session /path/to/session sync
    uv run python -m src.arena_render.run_arena_render --session /path/to/session rerun
    uv run python -m src.arena_render.run_arena_render --session /path/to/session extract-textures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.arena_render.arena_config import ArenaRenderGeometry, default_geometry_template_path
from src.arena_render.pipeline_config import ArenaRenderConfig, default_config_path, load_arena_render_config
from src.arena_render.rerun_arena_viewer import launch_arena_rerun_viewer
from src.arena_render.session_inputs import resolve_arena_session_inputs
from src.arena_render.timeline_exporter import build_stimulus_timeline, write_stimulus_timeline
from src.arena_render.validate_arena import print_validation_report, validate_arena_session
from src.arena_render.wall_texture_extractor import extract_wall_textures_for_session
from src.cameras.postprocess import postprocess
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _default_output_dir(session_root: Path) -> Path:
	return session_root / "full_recording" / "arena_render"


def _resolve_geometry(config: ArenaRenderConfig, session_root: Path) -> ArenaRenderGeometry:
	if config.geometry_path is not None and config.geometry_path.is_file():
		return ArenaRenderGeometry.from_json_file(config.geometry_path)
	session_geometry = session_root / "arena_geometry.json"
	if session_geometry.is_file():
		return ArenaRenderGeometry.from_json_file(session_geometry)
	raise FileNotFoundError(
		f"No arena geometry JSON found. Copy {default_geometry_template_path()} "
		f"to {session_geometry} and set wall ROIs."
	)


def cmd_validate(config: ArenaRenderConfig) -> int:
	assert config.session_root is not None
	report = validate_arena_session(
		config.session_root,
		calibration_toml_path=config.calibration_toml_path,
	)
	print_validation_report(report)
	return 0 if report.ok else 1


def cmd_sync(config: ArenaRenderConfig) -> int:
	assert config.session_root is not None
	logger.info("Syncing overhead videos only (include_eyes=False)")
	postprocess(config.session_root.resolve(), include_eyes=False)
	return cmd_validate(config)


def cmd_rerun(config: ArenaRenderConfig, preview_frame: int | None) -> int:
	assert config.session_root is not None
	inputs = resolve_arena_session_inputs(
		config.session_root,
		calibration_toml_path=config.calibration_toml_path,
	)
	geometry = None
	texture_root = None
	try:
		geometry = _resolve_geometry(config, inputs.session_root)
	except FileNotFoundError as exc:
		logger.warning("%s — logging cameras and groundplane only", exc)
	output_dir = config.output_dir or _default_output_dir(inputs.session_root)
	if (output_dir / "wall_textures").is_dir():
		texture_root = output_dir / "wall_textures"
	launch_arena_rerun_viewer(
		inputs,
		geometry,
		texture_root=texture_root,
		preview_frame=preview_frame,
		spawn=True,
	)
	return 0


def cmd_extract_textures(config: ArenaRenderConfig) -> int:
	assert config.session_root is not None
	inputs = resolve_arena_session_inputs(
		config.session_root,
		calibration_toml_path=config.calibration_toml_path,
	)
	geometry = _resolve_geometry(config, inputs.session_root)
	if not geometry.walls:
		logger.error("arena geometry has no walls configured")
		return 1
	output_dir = config.output_dir or _default_output_dir(inputs.session_root)
	texture_root = output_dir / "wall_textures"
	wall_paths = extract_wall_textures_for_session(
		inputs,
		geometry,
		texture_root,
		frame_stride=config.frame_stride,
		max_frames=config.max_frames,
	)
	timeline = build_stimulus_timeline(
		inputs,
		geometry,
		texture_root,
		wall_paths,
		frame_stride=config.frame_stride,
	)
	timeline_path = output_dir / "stimulus_timeline.json"
	write_stimulus_timeline(timeline, timeline_path)
	logger.info("Wrote %s", timeline_path)
	return 0


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Offline arena virtual render pipeline")
	parser.add_argument("--config", type=Path, default=None, help="JSON config path")
	parser.add_argument("--session", type=Path, default=None, help="Session root path")
	parser.add_argument(
		"--geometry",
		type=Path,
		default=None,
		help="Arena geometry JSON (wall ROIs and screen positions)",
	)
	parser.add_argument("--calibration-toml", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--frame-stride", type=int, default=None)
	parser.add_argument("--max-frames", type=int, default=None)
	parser.add_argument("--preview-frame", type=int, default=0, help="Frame for Rerun texture preview")
	parser.add_argument(
		"command",
		choices=("validate", "sync", "rerun", "extract-textures"),
		help="Pipeline step to run",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)
	config_path = args.config or default_config_path()
	config = load_arena_render_config(config_path if config_path.is_file() else None)
	if args.session is not None:
		config.session_root = args.session.resolve()
	if args.geometry is not None:
		config.geometry_path = args.geometry.resolve()
	if args.calibration_toml is not None:
		config.calibration_toml_path = args.calibration_toml.resolve()
	if args.output_dir is not None:
		config.output_dir = args.output_dir.resolve()
	if args.frame_stride is not None:
		config.frame_stride = args.frame_stride
	if args.max_frames is not None:
		config.max_frames = args.max_frames

	if config.session_root is None:
		logger.error("--session or session_root in config is required")
		return 1

	if args.command == "validate":
		return cmd_validate(config)
	if args.command == "sync":
		return cmd_sync(config)
	if args.command == "rerun":
		return cmd_rerun(config, preview_frame=args.preview_frame)
	if args.command == "extract-textures":
		return cmd_extract_textures(config)
	return 1


if __name__ == "__main__":
	sys.exit(main())
