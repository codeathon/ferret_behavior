"""
Discover arena-render inputs from a ferret session folder.

Arena-only scope: overhead sync videos + calibration TOML; no pupil or DLC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cameras.synchronization.time_units import resolve_synchronized_video_dir
from src.offline_pov.validate_session import resolve_session_paths
from src.utilities.folder_utilities.session_paths import discover_calibration_toml


@dataclass(frozen=True)
class SyncedOverheadVideo:
	"""One synchronized overhead Basler stream."""

	stem: str
	path: Path
	timestamps_utc_path: Path
	frame_count: int


@dataclass(frozen=True)
class ArenaSessionInputs:
	"""Resolved paths needed for arena sync, validation, and texture extraction."""

	session_root: Path
	full_recording: Path
	sync_video_dir: Path
	calibration_toml: Path
	overhead_videos: list[SyncedOverheadVideo]


def discover_sync_video_dir(session_root: Path) -> Path:
	"""Return synchronized overhead folder under ``full_recording/mocap_data``."""
	_, full_recording = resolve_session_paths(session_root)
	mocap_data = full_recording / "mocap_data"
	if not mocap_data.is_dir():
		raise FileNotFoundError(f"Missing mocap_data under {full_recording}")
	sync_dir = resolve_synchronized_video_dir(mocap_data)
	if not sync_dir.is_dir():
		raise FileNotFoundError(f"No synchronized video folder under {mocap_data}")
	return sync_dir


def _match_timestamps(sync_dir: Path, video_stem: str) -> Path:
	matches = sorted(sync_dir.glob(f"{video_stem}*_timestamps_utc.npy"))
	if not matches:
		# Stems are often serial-only before first underscore in timestamp glob.
		prefix = video_stem.split("_")[0]
		matches = sorted(sync_dir.glob(f"{prefix}*_timestamps_utc.npy"))
	if not matches:
		raise FileNotFoundError(f"No UTC timestamps for video stem {video_stem} in {sync_dir}")
	return matches[0]


def list_overhead_videos(sync_dir: Path) -> list[SyncedOverheadVideo]:
	"""List overhead mp4 streams, excluding combined/rotated review outputs."""
	videos: list[SyncedOverheadVideo] = []
	for mp4 in sorted(sync_dir.glob("*.mp4")):
		stem = mp4.stem
		if "combined" in stem or "rotated" in stem or "eye" in stem:
			continue
		ts_path = _match_timestamps(sync_dir, stem)
		timestamps = np.load(ts_path, allow_pickle=True)
		videos.append(
			SyncedOverheadVideo(
				stem=stem,
				path=mp4,
				timestamps_utc_path=ts_path,
				frame_count=len(timestamps),
			)
		)
	return videos


def resolve_arena_session_inputs(
	session_root: Path,
	calibration_toml_path: Path | None = None,
) -> ArenaSessionInputs:
	"""Resolve sync videos and calibration TOML for arena rendering."""
	session_root = session_root.resolve()
	session_root, full_recording = resolve_session_paths(session_root)
	sync_dir = discover_sync_video_dir(session_root)
	toml = discover_calibration_toml(session_root, explicit_path=calibration_toml_path)
	if toml is None:
		raise FileNotFoundError(
			f"No *camera_calibration.toml under {session_root}. "
			"Run calibration or set calibration_toml_path."
		)
	overhead = list_overhead_videos(sync_dir)
	if not overhead:
		raise FileNotFoundError(f"No overhead mp4 files in {sync_dir}")
	return ArenaSessionInputs(
		session_root=session_root,
		full_recording=full_recording,
		sync_video_dir=sync_dir,
		calibration_toml=toml,
		overhead_videos=overhead,
	)


def find_video_by_camera_name(
	overhead_videos: list[SyncedOverheadVideo],
	camera_name: str,
) -> SyncedOverheadVideo:
	"""Match a wall's ``source_camera_name`` to a synchronized mp4 stem."""
	matches = [video for video in overhead_videos if camera_name in video.stem]
	if not matches:
		names = ", ".join(video.stem for video in overhead_videos)
		raise KeyError(f"Camera {camera_name!r} not found in synced videos: {names}")
	if len(matches) > 1:
		# Prefer exact stem prefix match when multiple cameras share a substring.
		exact = [video for video in matches if video.stem.startswith(camera_name)]
		if exact:
			return exact[0]
	return matches[0]
