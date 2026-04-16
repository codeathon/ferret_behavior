"""
Wall UTC bridge for aligning Pupil device time with Basler grab latch timeline.

``PupilClockMapper.pupil_to_host_utc_ns`` maps Pupil time into the same numeric
domain as ``time.monotonic_ns()`` samples used when fitting the mapper (not
literal Unix epoch). This module converts that estimate to wall UTC using the
Basler latch :class:`~src.cameras.diagnostics.timestamp_mapping.TimestampMapping`
so Pupil frames can be compared to ``BaslerFrameSet.anchor_utc_ns``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.cameras.diagnostics.timestamp_mapping import TimestampMapping
from src.cameras.synchronization.pupil_clock_sync import PupilClockMapper


@dataclass
class WallUtcFromPupilTime:
    """
    Map Pupil timeline (ns) to wall UTC (ns) aligned with a Basler grab latch.

    Uses ``latch.utc_time_ns + (host_mono_from_mapper - latch.monotonic_ns)`` so
    Pupil instants line up with
    :func:`~src.cameras.timestamp_utils.basler_frame_utc_ns_from_latch_delta`
    for the same recording session.
    """

    latch: TimestampMapping
    mapper: PupilClockMapper

    def pupil_time_ns_to_wall_utc_ns(self, pupil_time_ns: int) -> int:
        # Mapper output is in the same space as ClockSample.host_monotonic_ns.
        host_mono_est = self.mapper.pupil_to_host_utc_ns(pupil_time_ns)
        return int(self.latch.utc_time_ns + (host_mono_est - self.latch.monotonic_ns))
