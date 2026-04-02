"""
Shared time-unit helpers for camera synchronization modules.

Provides unit conversion and the synchronized-video-directory resolver
used by TimestampConverter and PupilSynchronize.
"""

from pathlib import Path


def seconds_to_nanoseconds(seconds: float) -> int:
    """Convert seconds to integer nanoseconds."""
    return int(seconds * 1e9)


def nanoseconds_to_seconds(nanoseconds: int) -> float:
    """Convert nanoseconds to seconds."""
    return nanoseconds / 1e9


def resolve_synchronized_video_dir(folder_path: Path) -> Path:
    """
    Return the synchronized video directory for a recording folder.

    Prefers 'synchronized_corrected_videos' (post-correction) and falls
    back to 'synchronized_videos' if the corrected directory is absent.
    """
    corrected = folder_path / "synchronized_corrected_videos"
    if corrected.exists():
        return corrected
    return folder_path / "synchronized_videos"
