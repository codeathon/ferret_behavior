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


def _validate_calibration_toml(report: SessionValidationReport, toml_path: Path | None) -> None:
	if toml_path is None or not toml_path.is_file():
		report.add(
			"calibration_toml",
			ValidationLevel.ERROR,
			"no *camera_calibration.toml under session/calibration/",
			toml_path,
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


def _validate_raw_inputs(report: SessionValidationReport, recording: RecordingFolder) -> None:
	mocap_raw = recording.mocap_data / "raw_videos"
	if _check_path_exists(report, "mocap_raw_videos", mocap_raw if mocap_raw.exists() else None):
		ts_map = mocap_raw / "timestamp_mapping.json"
		_check_path_exists(report, "basler_timestamp_mapping", ts_map if ts_map.exists() else None)

	eye_videos = recording.eye_videos
	if _check_path_exists(report, "eye_videos", eye_videos):
		for label, path in {
			"left_eye_video": recording.left_eye_video,
			"right_eye_video": recording.right_eye_video,
			"pupil_world_video": recording.pupil_world_video,
		}.items():
			_check_path_exists(report, label, path)

	pupil_info = recording.eye_data / "info.player.json"
	if not pupil_info.exists():
		pupil_export = recording.base_recordings_folder / "pupil_output" / "info.player.json"
		pupil_info = pupil_export if pupil_export.exists() else pupil_info
	if pupil_info.exists():
		report.add("pupil_timestamp_mapping", ValidationLevel.INFO, "found", pupil_info)
	else:
		report.add(
			"pupil_timestamp_mapping",
			ValidationLevel.WARN,
			"info.player.json not found (needed if sync step not yet run)",
			pupil_info,
		)


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
	_validate_calibration_toml(report, recording.calibration_toml_path)
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
