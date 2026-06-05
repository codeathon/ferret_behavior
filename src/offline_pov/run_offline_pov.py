"""
CLI entrypoint for the offline ferret POV pipeline.

Usage:
    uv run python -m src.offline_pov.run_offline_pov --session /path/to/session --validate-only
    uv run python -m src.offline_pov.run_offline_pov --config configs/offline_pipeline.json
    uv run python -m src.offline_pov.run_offline_pov --session /path/to/session --clip 0m_37s-1m_37s
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.batch_processing.full_pipeline import run_pipeline
from src.ferret_gaze.run_gaze_pipeline import run_gaze_pipeline
from src.offline_pov.pipeline_config import (
	OfflinePipelineConfig,
	default_config_path,
	load_offline_pipeline_config,
)
from src.offline_pov.validate_session import (
	print_validation_report,
	resolve_session_paths,
	validate_session,
)
from src.utilities.folder_utilities.session_paths import discover_calibration_toml
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _apply_external_tool_paths(config: OfflinePipelineConfig) -> None:
	"""Wire config paths into full_pipeline subprocess helpers."""
	from src.batch_processing.full_pipeline import ExternalToolPaths, set_external_tool_paths

	tools = config.external_tools
	set_external_tool_paths(
		ExternalToolPaths(
			skellyclicker_python=tools.skellyclicker_python,
			skellyclicker_script=tools.skellyclicker_script,
			triangulation_python=tools.triangulation_python,
			triangulation_script=tools.triangulation_script,
			calibration_python=tools.calibration_python,
			calibration_script=tools.calibration_script,
		)
	)


def _resolve_recording_folder(session_input: Path) -> Path:
	"""Return ``full_recording`` or clip path for pipeline execution."""
	session_input = session_input.resolve()
	if "clips" in session_input.parts:
		return session_input
	_, full_recording = resolve_session_paths(session_input)
	return full_recording


def _resolve_clip_path(session_root: Path, clip_name: str) -> Path:
	clip_path = session_root / "clips" / clip_name
	if not clip_path.is_dir():
		raise FileNotFoundError(f"Clip not found: {clip_path}")
	return clip_path


def run_validate(
	config: OfflinePipelineConfig,
	*,
	check_calibrated: bool = False,
	require_calibration_toml: bool = False,
) -> bool:
	session_path = config.session_root
	assert session_path is not None
	report = validate_session(
		session_path,
		calibration_toml_path=config.calibration_toml_path,
		require_calibration_toml=require_calibration_toml,
		check_calibrated=check_calibrated,
	)
	print_validation_report(report)
	return report.ok


def run_full_offline_pipeline(config: OfflinePipelineConfig, recording_folder: Path) -> None:
	_apply_external_tool_paths(config)
	logger.info("Running offline batch pipeline on %s", recording_folder)
	run_pipeline(
		recording_folder_path=recording_folder,
		calibration_toml_path=config.calibration_toml_path,
		include_eye=config.include_eye,
		overwrite_synchronization=config.overwrite_synchronization,
		overwrite_calibration=config.overwrite_calibration,
		overwrite_dlc=config.overwrite_dlc,
		overwrite_triangulation=config.overwrite_triangulation,
		overwrite_eye_postprocessing=config.overwrite_eye_postprocessing,
		overwrite_skull_postprocessing=config.overwrite_skull_postprocessing,
		overwrite_gaze=config.overwrite_gaze,
		mode="offline",
	)


def run_gaze_and_blender(config: OfflinePipelineConfig, clip_path: Path) -> Path:
	logger.info("Running gaze pipeline on clip %s", clip_path)
	output_dir = run_gaze_pipeline(
		recording_path=clip_path,
		reprocess_all=config.reprocess_gaze_clip,
	)
	logger.info("Gaze pipeline complete: %s", output_dir)
	return output_dir


def copy_clip_for_local_inspection(clip_path: Path, destination: Path) -> Path:
	"""
	Copy ``analyzable_output/`` and ``display_videos/`` from a processed clip.

	Returns the destination root created under ``destination``.
	"""
	destination = destination.resolve()
	destination.mkdir(parents=True, exist_ok=True)
	clip_name = clip_path.name
	target = destination / clip_name
	if target.exists():
		shutil.rmtree(target)
	target.mkdir(parents=True)

	for subdir in ("analyzable_output", "display_videos"):
		src = clip_path / subdir
		if src.is_dir():
			shutil.copytree(src, target / subdir)
			logger.info("Copied %s -> %s", src, target / subdir)
		else:
			logger.warning("Skipping missing %s", src)

	readme = target / "README.txt"
	readme.write_text(
		"Ferret POV inspection bundle\n\n"
		"Blender: open analyzable_output/*blender*.py in Blender 4.0+, Alt+P, Spacebar to play.\n"
		"Rerun: uv run python src/rerun_viewer/everything_viewer.py (update paths in script).\n",
		encoding="utf-8",
	)
	logger.info("Local inspection bundle ready at %s", target)
	return target


def run_from_config(
	config: OfflinePipelineConfig,
	*,
	validate_only: bool = False,
	skip_pipeline: bool = False,
	skip_gaze: bool = False,
	skip_copy: bool = False,
) -> int:
	if config.session_root is None:
		logger.error("session_root is required in config or --session")
		return 1

	session_root, _ = resolve_session_paths(config.session_root)
	if not run_validate(config, require_calibration_toml=False):
		logger.error("Session validation failed; fix errors before running pipeline")
		return 1
	if validate_only:
		return 0

	# Triangulation needs a calibration TOML; trial sessions often borrow one from another session.
	if not skip_pipeline and "clips" not in _resolve_recording_folder(config.session_root).parts:
		resolved_toml = discover_calibration_toml(session_root, config.calibration_toml_path)
		if resolved_toml is None:
			logger.error(
				"No *camera_calibration.toml found. Set calibration_toml_path in "
				"configs/offline_pipeline.json to a prior calibrated session, e.g. "
				"/home/scholl-lab/ferret_recordings/session_YYYY-MM-DD_.../calibration/"
				"session_*_camera_calibration.toml"
			)
			return 1
		config.calibration_toml_path = resolved_toml
		if not run_validate(config, require_calibration_toml=True):
			logger.error("Calibration TOML validation failed")
			return 1

	recording_folder = _resolve_recording_folder(config.session_root)
	if not skip_pipeline and "clips" not in recording_folder.parts:
		run_full_offline_pipeline(config, recording_folder)

	if config.clip_name:
		clip_path = _resolve_clip_path(session_root, config.clip_name)
		if not skip_gaze:
			run_gaze_and_blender(config, clip_path)
		if not skip_copy:
			copy_clip_for_local_inspection(clip_path, config.local_inspection_dir)

	return 0


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Offline ferret POV pipeline runner")
	parser.add_argument(
		"--config",
		type=Path,
		default=None,
		help=f"JSON config path (default: {default_config_path()})",
	)
	parser.add_argument("--session", type=Path, default=None, help="Session root or full_recording path")
	parser.add_argument("--clip", type=str, default=None, help="Clip name under session/clips/")
	parser.add_argument("--validate-only", action="store_true", help="Only validate session layout")
	parser.add_argument(
		"--calibration-toml",
		type=Path,
		default=None,
		help="Reuse *camera_calibration.toml from another session (trial folders often have none)",
	)
	parser.add_argument(
		"--require-calibration",
		action="store_true",
		help="Treat missing calibration TOML as an error during --validate-only",
	)
	parser.add_argument("--check-calibrated", action="store_true", help="Also require calibration outputs")
	parser.add_argument("--skip-pipeline", action="store_true", help="Skip full_pipeline batch step")
	parser.add_argument("--skip-gaze", action="store_true", help="Skip gaze + Blender script generation")
	parser.add_argument("--skip-copy", action="store_true", help="Skip copy to local_inspection_dir")
	parser.add_argument(
		"--local-inspection-dir",
		type=Path,
		default=None,
		help="Destination for analyzable_output copy (default: ~/ferret_pov_inspection)",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)
	config_path = args.config or default_config_path()
	config = load_offline_pipeline_config(config_path if config_path.is_file() else None)

	if args.session is not None:
		config.session_root = args.session.resolve()
	if args.clip is not None:
		config.clip_name = args.clip
	if args.calibration_toml is not None:
		config.calibration_toml_path = args.calibration_toml.resolve()
	if args.local_inspection_dir is not None:
		config.local_inspection_dir = args.local_inspection_dir.resolve()

	if args.validate_only and config.session_root is None:
		logger.error("--session is required for --validate-only")
		return 1
	if args.validate_only:
		ok = run_validate(
			config,
			check_calibrated=args.check_calibrated,
			require_calibration_toml=args.require_calibration,
		)
		return 0 if ok else 1

	return run_from_config(
		config,
		validate_only=False,
		skip_pipeline=args.skip_pipeline,
		skip_gaze=args.skip_gaze,
		skip_copy=args.skip_copy,
	)


if __name__ == "__main__":
	sys.exit(main())
