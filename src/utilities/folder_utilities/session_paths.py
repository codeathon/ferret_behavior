"""
Shared session path discovery for ferret recording folders.

Canonical capture layout (used by postprocess/sync):
    session/base_data/raw_videos/
    session/base_data/pupil_output/info.player.json
"""

from __future__ import annotations

from pathlib import Path

PUPIL_INFO_CANDIDATE_NAMES = ("info.player.json", "info.json")
CALIBRATION_TOML_GLOB = "*camera_calibration.toml"


def session_base_data_folder(session_root: Path) -> Path:
	"""Return ``session_root/base_data``."""
	return session_root / "base_data"


def discover_pupil_output_folder(session_root: Path) -> Path | None:
	"""
	Locate Pupil export directory under a session.

	Preferred: ``base_data/pupil_output`` (current layout).
	Fallback: ``pupil_output`` at session root (legacy).
	"""
	candidates = [
		session_base_data_folder(session_root) / "pupil_output",
		session_root / "pupil_output",
	]
	for path in candidates:
		if path.is_dir():
			return path
	return None


def discover_pupil_info_json(session_root: Path) -> Path | None:
	"""
	Locate Pupil timestamp metadata (``info.player.json`` or ``info.json``).

	Searches ``base_data/pupil_output`` first, then legacy locations.
	"""
	pupil_output = discover_pupil_output_folder(session_root)
	if pupil_output is not None:
		for name in PUPIL_INFO_CANDIDATE_NAMES:
			candidate = pupil_output / name
			if candidate.is_file():
				return candidate

	full_recording = session_root / "full_recording"
	for relative in (
		Path("eye_data") / "info.player.json",
		Path("eye_data") / "info.json",
		Path("eye_data") / "eye_videos" / "info.player.json",
	):
		candidate = full_recording / relative
		if candidate.is_file():
			return candidate
	return None


def discover_calibration_toml(
	session_root: Path,
	explicit_path: Path | None = None,
) -> Path | None:
	"""
	Locate multi-camera calibration TOML for a session.

	Trial folders often have no TOML; pass ``explicit_path`` to reuse calibration
	from an earlier session on the same rig.

	Search order:
	1. ``explicit_path`` if provided and exists
	2. ``session/calibration/*camera_calibration.toml``
	3. Any ``*camera_calibration.toml`` under ``session_root`` (recursive)
	"""
	if explicit_path is not None and explicit_path.is_file():
		return explicit_path.resolve()

	cal_dir = session_root / "calibration"
	if cal_dir.is_dir():
		matches = sorted(cal_dir.glob(CALIBRATION_TOML_GLOB))
		if matches:
			return matches[0]

	matches = sorted(session_root.rglob(CALIBRATION_TOML_GLOB))
	return matches[0] if matches else None


def session_has_calibration_videos(session_root: Path) -> bool:
	"""Return True if ``session/calibration`` contains mp4s usable for Charuco calibration."""
	cal_dir = session_root / "calibration"
	if not cal_dir.is_dir():
		return False
	return any(cal_dir.rglob("*.mp4"))


def discover_basler_raw_videos_folder(session_root: Path) -> Path | None:
	"""
	Locate Basler raw capture folder.

	Preferred: ``base_data/raw_videos``. Fallback: ``full_recording/mocap_data/raw_videos``.
	"""
	candidates = [
		session_base_data_folder(session_root) / "raw_videos",
		session_root / "full_recording" / "mocap_data" / "raw_videos",
	]
	for path in candidates:
		if path.is_dir():
			return path
	return None
