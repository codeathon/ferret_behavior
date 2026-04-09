"""
Pupil clock synchronization scaffolding for realtime pipelines.

Step 5 goal: provide a direct host-monotonic <-> Pupil-time mapping surface that
can later be backed by live Pupil ZMQ samples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClockSample:
    """One synchronized sample pair between host and Pupil clocks."""

    host_monotonic_ns: int
    pupil_time_ns: int


class PupilClockMapper:
    """Linear host<->Pupil clock mapper with drift/offset terms."""

    def __init__(self, offset_ns: int = 0, drift_ppm: float = 0.0) -> None:
        self._offset_ns = offset_ns
        self._drift_ppm = drift_ppm

    @property
    def offset_ns(self) -> int:
        return self._offset_ns

    @property
    def drift_ppm(self) -> float:
        return self._drift_ppm

    def pupil_to_host_utc_ns(self, pupil_time_ns: int) -> int:
        """
        Map Pupil time to host UTC-like nanoseconds.

        drift_ppm is applied as multiplicative scale around 1.0.
        """
        scale = 1.0 + (self._drift_ppm / 1_000_000.0)
        return int((pupil_time_ns * scale) + self._offset_ns)

    def host_to_pupil_time_ns(self, host_monotonic_ns: int) -> int:
        """Inverse map from host monotonic ns to Pupil time ns."""
        scale = 1.0 + (self._drift_ppm / 1_000_000.0)
        return int((host_monotonic_ns - self._offset_ns) / scale)

    def update_from_samples(self, first: ClockSample, second: ClockSample) -> None:
        """Estimate linear map params from two sample pairs."""
        host_delta = second.host_monotonic_ns - first.host_monotonic_ns
        pupil_delta = second.pupil_time_ns - first.pupil_time_ns
        if host_delta <= 0 or pupil_delta <= 0:
            raise ValueError("Clock samples must have positive deltas")
        scale = host_delta / pupil_delta
        self._drift_ppm = (scale - 1.0) * 1_000_000.0
        self._offset_ns = int(first.host_monotonic_ns - (first.pupil_time_ns * scale))
