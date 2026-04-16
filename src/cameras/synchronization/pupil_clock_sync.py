"""
Pupil clock synchronization utilities for realtime pipelines.

Provides a linear host-monotonic <-> Pupil-time mapper and optional live sample
collection through the Pupil Remote ZMQ API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


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
        Map Pupil time into the same numeric domain as ``ClockSample.host_monotonic_ns``.

        Despite the name, this is **not** wall Unix UTC; pair with
        :class:`~src.cameras.synchronization.utc_clock_bridge.WallUtcFromPupilTime`
        to compare against Basler ``capture_utc_ns``. ``drift_ppm`` scales Pupil time
        around 1.0 before applying ``offset_ns``.
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

    @classmethod
    def from_live_samples(
        cls,
        endpoint: str = "tcp://127.0.0.1:50020",
        n_samples: int = 8,
        settle_seconds: float = 0.02,
        timeout_ms: int = 500,
    ) -> "PupilClockMapper":
        """
        Build a mapper from live Pupil Remote clock queries over ZMQ REQ/REP.

        The request payload is the single token ``"t"`` (Pupil time in seconds).
        Host timestamps are measured as midpoint monotonic time around each round
        trip to reduce transport asymmetry error.
        """
        samples = collect_live_clock_samples(
            endpoint=endpoint,
            n_samples=n_samples,
            settle_seconds=settle_seconds,
            timeout_ms=timeout_ms,
        )
        return cls.from_sample_series(samples)

    @classmethod
    def from_sample_series(cls, samples: list[ClockSample]) -> "PupilClockMapper":
        """
        Fit mapper parameters from >=2 clock samples with endpoint regression.
        """
        if len(samples) < 2:
            raise ValueError("At least two clock samples are required")
        mapper = cls()
        mapper.update_from_samples(samples[0], samples[-1])
        return mapper


def collect_live_clock_samples(
    endpoint: str = "tcp://127.0.0.1:50020",
    n_samples: int = 8,
    settle_seconds: float = 0.02,
    timeout_ms: int = 500,
) -> list[ClockSample]:
    """
    Collect host/Pupil clock sample pairs from Pupil Remote over ZMQ.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if settle_seconds < 0:
        raise ValueError("settle_seconds cannot be negative")

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("Pupil clock sync requires pyzmq. Install with: uv add pyzmq") from exc

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(endpoint)

    samples: list[ClockSample] = []
    try:
        for _ in range(n_samples):
            host_send_ns = time.monotonic_ns()
            socket.send_string("t")
            response = socket.recv_string()
            host_recv_ns = time.monotonic_ns()
            host_midpoint_ns = host_send_ns + ((host_recv_ns - host_send_ns) // 2)

            pupil_seconds = float(response.strip())
            pupil_time_ns = int(pupil_seconds * 1_000_000_000.0)
            samples.append(
                ClockSample(host_monotonic_ns=host_midpoint_ns, pupil_time_ns=pupil_time_ns)
            )
            if settle_seconds > 0:
                time.sleep(settle_seconds)
    finally:
        socket.close(linger=0)

    logger.info("Collected %d Pupil clock samples from %s", len(samples), endpoint)
    return samples
