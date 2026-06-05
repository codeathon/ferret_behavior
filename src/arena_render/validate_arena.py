"""
Validate session readiness for arena-only virtual rendering.

Checks overhead sync videos and calibration TOML; does not require pupil or DLC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.arena_render.session_inputs import (
	discover_sync_video_dir,
	list_overhead_videos,
	resolve_arena_session_inputs,
)
from src.ferret_gaze.realtime.calibration_projection import load_session_multi_view_calibration
from src.offline_pov.validate_session import resolve_session_paths
from src.utilities.folder_utilities.session_paths import discover_calibration_toml

REQUIRED_CALIBRATION_CAM_COUNT = 5


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
class ArenaValidationReport:
	session_root: Path
	items: list[ValidationItem] = field(default_factory=list)

	@property
	def ok(self) -> bool:
		return not any(item.level == ValidationLevel.ERROR for item in self.items)

	def add(self, name: str, level: ValidationLevel, message: str, path: Path | None = None) -> None:
		self.items.append(ValidationItem(name=name, level=level, message=message, path=path))

	def summary_lines(self) -> list[str]:
		lines = [f"Arena validation: {self.session_root}", f"Status: {'PASS' if self.ok else 'FAIL'}"]
		for item in self.items:
			prefix = item.level.value.upper()
			loc = f" ({item.path})" if item.path else ""
			lines.append(f"  [{prefix}] {item.name}: {item.message}{loc}")
		return lines


def validate_arena_session(
	session_root: Path,
	calibration_toml_path: Path | None = None,
) -> ArenaValidationReport:
	"""Check inputs required for arena reconstruction (no ferret tracking)."""
	session_root = session_root.resolve()
	session_root, full_recording = resolve_session_paths(session_root)
	report = ArenaValidationReport(session_root=session_root)

	toml = discover_calibration_toml(session_root, explicit_path=calibration_toml_path)
	if toml is None:
		report.add(
			"calibration_toml",
			ValidationLevel.ERROR,
			"Missing *camera_calibration.toml",
			session_root / "calibration",
		)
	else:
		report.add(
			"calibration_toml",
			ValidationLevel.INFO,
			f"Found calibration TOML ({toml.name})",
			toml,
		)
		try:
			calib = load_session_multi_view_calibration(toml)
			count = len(calib.projection_by_cam_index)
			if count < REQUIRED_CALIBRATION_CAM_COUNT:
				report.add(
					"calibration_cameras",
					ValidationLevel.WARN,
					f"Expected {REQUIRED_CALIBRATION_CAM_COUNT} cameras, found {count}",
					toml,
				)
			else:
				report.add(
					"calibration_cameras",
					ValidationLevel.INFO,
					f"{count} camera projection matrices loaded",
					toml,
				)
		except Exception as exc:
			report.add("calibration_parse", ValidationLevel.ERROR, str(exc), toml)

	try:
		sync_dir = discover_sync_video_dir(session_root)
		report.add("sync_videos", ValidationLevel.INFO, f"Sync folder: {sync_dir.name}", sync_dir)
		overhead = list_overhead_videos(sync_dir)
		report.add(
			"overhead_streams",
			ValidationLevel.INFO if overhead else ValidationLevel.ERROR,
			f"{len(overhead)} overhead mp4 streams",
			sync_dir,
		)
		for video in overhead:
			report.add(
				f"video_{video.stem}",
				ValidationLevel.INFO,
				f"{video.frame_count} frames",
				video.path,
			)
	except FileNotFoundError as exc:
		report.add("sync_videos", ValidationLevel.ERROR, str(exc), full_recording / "mocap_data")

	return report


def print_validation_report(report: ArenaValidationReport) -> None:
	for line in report.summary_lines():
		print(line)


def require_arena_inputs(
	session_root: Path,
	calibration_toml_path: Path | None = None,
):
	"""Validate then return resolved inputs, raising on failure."""
	report = validate_arena_session(session_root, calibration_toml_path=calibration_toml_path)
	print_validation_report(report)
	if not report.ok:
		raise RuntimeError("Arena session validation failed")
	return resolve_arena_session_inputs(session_root, calibration_toml_path=calibration_toml_path)
