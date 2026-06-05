"""
Validate ferret session folders before running the offline POV pipeline.

Checks raw inputs (videos, timestamp sidecars, calibration TOML) and optional
pipeline stage completion (sync, calibration, DLC, gaze).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import toml

from src.ferret_gaze.realtime.calibration_projection import load_session_multi_view_calibration
from src.utilities.folder_utilities.calibration_folder import (
	CalibrationFolder,
	CalibrationPipelineStep,
)
from src.utilities.folder_utilities.recording_folder import BaslerCamera, RecordingFolder
from src.utilities.folder_utilities.session_paths import (
	discover_basler_raw_videos_folder,
	discover_calibration_toml,
	discover_pupil_info_json,
	discover_pupil_output_folder,
	session_has_calibration_videos,
)
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_CALIBRATION_CAM_COUNT = 5
REQUIRED_CALIBRATION_FIELDS = ("matrix", "world_position", "world_orientation")


class ValidationLevel(str, Enum):
	INFO = "info"
	WARN = "warn"
	ERROR = "error"


@dataclass
class ValidationItem:
	name: str
	level: ValidationLevel
	message: str
	path: Path | None = None


@dataclass
class SessionValidationReport:
	session_root: Path
	items: list[ValidationItem] = field(default_factory=list)

	@property
	def ok(self) -> bool:
		return not any(item.level == ValidationLevel.ERROR for item in self.items)

	@property
	def errors(self) -> list[ValidationItem]:
		return [item for item in self.items if item.level == ValidationLevel.ERROR]

	@property
	def warnings(self) -> list[ValidationItem]:
		return [item for item in self.items if item.level == ValidationLevel.WARN]

	def add(self, name: str, level: ValidationLevel, message: str, path: Path | None = None) -> None:
		self.items.append(ValidationItem(name=name, level=level, message=message, path=path))

	def summary_lines(self) -> list[str]:
		lines = [f"Session validation: {self.session_root}", f"Status: {'PASS' if self.ok else 'FAIL'}"]
		for item in self.items:
			prefix = item.level.value.upper()
			loc = f" ({item.path})" if item.path else ""
			lines.append(f"  [{prefix}] {item.name}: {item.message}{loc}")
		return lines


def resolve_session_paths(session_input: Path) -> tuple[Path, Path]:
	"""
	Normalize user input to ``(session_root, full_recording_path)``.

	Accepts session root, ``full_recording``, or a clip path under ``clips/``.
	"""
	session_input = session_input.resolve()
	if "clips" in session_input.parts:
		session_root = session_input.parent.parent
		full_recording = session_root / "full_recording"
		return session_root, full_recording
	if session_input.name == "full_recording":
		return session_input.parent, session_input
	if (session_input / "full_recording").is_dir():
		return session_input, session_input / "full_recording"
	if (session_input / "mocap_data").is_dir() and (session_input / "eye_data").is_dir():
		return session_input.parent, session_input
	raise ValueError(
		f"Could not resolve session layout from: {session_input}. "
		"Expected session root, full_recording, or clips/<name> path."
	)


def _check_path_exists(report: SessionValidationReport, name: str, path: Path | None) -> bool:
	if path is None or not path.exists():
		report.add(name, ValidationLevel.ERROR, "missing", path)
		return False
	report.add(name, ValidationLevel.INFO, "found", path)
	return True


def _missing_calibration_message(session_root: Path) -> str:
	if session_has_calibration_videos(session_root):
		return (
			"calibration videos found but no *camera_calibration.toml yet — "
			"run full_pipeline with overwrite_calibration=True, or set calibration_toml_path "
			"in configs/offline_pipeline.json to a prior session on this rig"
		)
	return (
		"no *camera_calibration.toml in this session (normal for trial-only folders) — "
		"set calibration_toml_path in configs/offline_pipeline.json to a calibrated session, "
		"or add session/calibration/ videos and run the calibration step"
	)


def _validate_calibration_toml(
	report: SessionValidationReport,
	session_root: Path,
	toml_path: Path | None,
	*,
	require_for_triangulation: bool,
) -> None:
	if toml_path is None or not toml_path.is_file():
		level = ValidationLevel.ERROR if require_for_triangulation else ValidationLevel.WARN
		report.add(
			"calibration_toml",
			level,
			_missing_calibration_message(session_root),
			session_root / "calibration",
		)
		return

	report.add("calibration_toml", ValidationLevel.INFO, "found", toml_path)
	raw = toml.load(toml_path)
	cam_keys = sorted(
		[k for k in raw if isinstance(k, str) and k.startswith("cam_")],
		key=lambda name: int(name.split("_", 1)[1]),
	)
	if len(cam_keys) < REQUIRED_CALIBRATION_CAM_COUNT:
		report.add(
			"calibration_cam_count",
			ValidationLevel.ERROR,
			f"expected >={REQUIRED_CALIBRATION_CAM_COUNT} cam_* blocks, found {len(cam_keys)}",
			toml_path,
		)
	else:
		report.add(
			"calibration_cam_count",
			ValidationLevel.INFO,
			f"found {len(cam_keys)} camera blocks",
			toml_path,
		)

	for key in cam_keys:
		block = raw.get(key, {})
		missing = [field_name for field_name in REQUIRED_CALIBRATION_FIELDS if field_name not in block]
		if missing:
			report.add(
				f"calibration_{key}",
				ValidationLevel.ERROR,
				f"missing fields: {', '.join(missing)}",
				toml_path,
			)

	try:
		load_session_multi_view_calibration(toml_path)
		report.add("calibration_projection", ValidationLevel.INFO, "projection matrices load OK", toml_path)
	except (ValueError, KeyError, TypeError) as exc:
		report.add(
			"calibration_projection",
			ValidationLevel.ERROR,
			f"failed to build projection matrices: {exc}",
			toml_path,
		)


def _pupil_video_path(pupil_output: Path, name: str) -> Path | None:
	"""Return first existing ``name*.mp4`` under a pupil export folder."""
	exact = pupil_output / f"{name}.mp4"
	if exact.is_file():
		return exact
	matches = sorted(pupil_output.glob(f"{name}*.mp4"))
	return matches[0] if matches else None


def _validate_raw_inputs(report: SessionValidationReport, recording: RecordingFolder) -> None:
	session_root = recording.base_recordings_folder

	# Basler raw capture lives under base_data/raw_videos in the current layout.
	mocap_raw = discover_basler_raw_videos_folder(session_root)
	if _check_path_exists(report, "mocap_raw_videos", mocap_raw):
		ts_map = mocap_raw / "timestamp_mapping.json"
		_check_path_exists(report, "basler_timestamp_mapping", ts_map if ts_map.exists() else None)

	# Pupil exports: canonical path is session/base_data/pupil_output.
	pupil_output = discover_pupil_output_folder(session_root)
	if pupil_output is not None:
		report.add("pupil_output_folder", ValidationLevel.INFO, "found", pupil_output)
	else:
		report.add(
			"pupil_output_folder",
			ValidationLevel.WARN,
			"base_data/pupil_output not found (expected before sync)",
			session_root / "base_data" / "pupil_output",
		)

	pupil_info = discover_pupil_info_json(session_root)
	if pupil_info is not None:
		report.add("pupil_timestamp_mapping", ValidationLevel.INFO, "found", pupil_info)
	else:
		report.add(
			"pupil_timestamp_mapping",
			ValidationLevel.WARN,
			"info.player.json / info.json not found under base_data/pupil_output",
			session_root / "base_data" / "pupil_output" / "info.player.json",
		)

	# Eye videos may be in full_recording (post-sync) or still in base_data/pupil_output.
	left_fallback = _pupil_video_path(pupil_output, recording.left_eye_name) if pupil_output else None
	right_fallback = _pupil_video_path(pupil_output, recording.right_eye_name) if pupil_output else None
	world_fallback = _pupil_video_path(pupil_output, "world") if pupil_output else None
	eye_sources: list[tuple[str, Path | None]] = [
		("left_eye_video", recording.left_eye_video or left_fallback),
		("right_eye_video", recording.right_eye_video or right_fallback),
		("pupil_world_video", recording.pupil_world_video or world_fallback),
	]

	for label, path in eye_sources:
		if path is not None and path.is_file():
			_check_path_exists(report, label, path)
		else:
			report.add(label, ValidationLevel.WARN, "not found in eye_videos or base_data/pupil_output", path)


def _validate_pipeline_stage(
	report: SessionValidationReport,
	recording: RecordingFolder,
	*,
	check_sync: bool,
	check_calibrated: bool,
) -> None:
	if check_sync:
		try:
			recording.check_synchronization()
			report.add("pipeline_sync", ValidationLevel.INFO, "synchronized videos present")
		except ValueError as exc:
			report.add("pipeline_sync", ValidationLevel.WARN, str(exc))

		for serial in [
			BaslerCamera.TOPDOWN.value,
			BaslerCamera.SIDE_0.value,
			BaslerCamera.SIDE_1.value,
			BaslerCamera.SIDE_2.value,
			BaslerCamera.SIDE_3.value,
		]:
			try:
				recording.get_synchronized_video_by_name(serial)
				recording.get_timestamp_by_name(serial)
			except ValueError:
				report.add(
					f"overhead_cam_{serial}",
					ValidationLevel.WARN,
					"synchronized video or timestamp missing",
					recording.mocap_synchronized_videos,
				)

	if check_calibrated:
		calibration_dir = recording.calibration_folder
		if calibration_dir is None:
			report.add("calibration_folder", ValidationLevel.ERROR, "session/calibration/ missing")
			return
		try:
			CalibrationFolder.from_folder_path(
				calibration_dir,
				expected_processing_step=CalibrationPipelineStep.CALIBRATED,
			)
			report.add("pipeline_calibration", ValidationLevel.INFO, "calibration outputs present")
		except ValueError as exc:
			report.add("pipeline_calibration", ValidationLevel.WARN, str(exc))


def validate_session(
	session_input: Path,
	*,
	calibration_toml_path: Path | None = None,
	require_calibration_toml: bool = False,
	check_sync: bool = True,
	check_calibrated: bool = False,
) -> SessionValidationReport:
	"""
	Validate a session for offline POV processing.

	Returns a report; raises only when the path cannot be resolved at all.
	"""
	session_root, full_recording = resolve_session_paths(session_input)
	report = SessionValidationReport(session_root=session_root)

	if not full_recording.is_dir():
		report.add("full_recording", ValidationLevel.ERROR, "missing", full_recording)
		return report

	try:
		recording = RecordingFolder.from_folder_path(full_recording)
	except ValueError as exc:
		report.add("recording_folder", ValidationLevel.ERROR, str(exc), full_recording)
		return report

	report.add("recording_folder", ValidationLevel.INFO, "layout OK", full_recording)
	_validate_raw_inputs(report, recording)
	resolved_toml = discover_calibration_toml(session_root, calibration_toml_path)
	_validate_calibration_toml(
		report,
		session_root,
		resolved_toml,
		require_for_triangulation=require_calibration_toml,
	)
	_validate_pipeline_stage(
		report,
		recording,
		check_sync=check_sync,
		check_calibrated=check_calibrated,
	)
	return report


def print_validation_report(report: SessionValidationReport) -> None:
	for line in report.summary_lines():
		logger.info(line)
