"""
Realtime transport publisher abstractions.

Step 2 scaffold:
- define a backend-agnostic publisher interface
- provide a no-op backend for local development/tests
- provide an optional ZMQ backend for Unreal integration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket
from src.utilities.logging_config import get_logger

logger = get_logger(__name__)


class RealtimePublisher(ABC):
    """Interface for realtime gaze packet publishing backends."""

    @abstractmethod
    def publish(self, packet: RealtimeGazePacket) -> None:
        """Publish one realtime gaze packet."""

    @abstractmethod
    def close(self) -> None:
        """Close resources owned by the publisher."""


class NoOpRealtimePublisher(RealtimePublisher):
    """No-op publisher used by scaffold mode and tests."""

    def publish(self, packet: RealtimeGazePacket) -> None:
        # Intentionally do nothing; useful for transport pipeline dry-runs.
        _ = packet

    def close(self) -> None:
        """No resources to close."""


class ZmqRealtimePublisher(RealtimePublisher):
    """
    ZMQ PUB backend for Unreal subscribers.

    This backend depends on `pyzmq` but imports it lazily so environments without
    pyzmq can still use offline mode and no-op realtime scaffolding.
    """

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556", topic: str = "gaze.live") -> None:
        try:
            import zmq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ZMQ realtime backend requires pyzmq. Install with: uv add pyzmq"
            ) from exc

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(endpoint)
        self._topic_bytes = topic.encode("utf-8")
        logger.info("Realtime ZMQ publisher bound to %s (topic=%s)", endpoint, topic)

    def publish(self, packet: RealtimeGazePacket) -> None:
        """
        Publish packet bytes as multipart [topic, payload].

        JSON is used in the scaffold to avoid coupling transport bring-up to a
        msgpack dependency decision. We can switch to msgpack in step 3+.
        """
        payload_bytes = packet.model_dump_json().encode("utf-8")
        self._socket.send_multipart([self._topic_bytes, payload_bytes])

    def close(self) -> None:
        """Close the PUB socket."""
        self._socket.close(linger=0)


def create_realtime_publisher(
    backend: str = "noop",
    **kwargs: Any,
) -> RealtimePublisher:
    """Factory for realtime publisher backends."""
    normalized = backend.strip().lower()
    if normalized == "noop":
        return NoOpRealtimePublisher()
    if normalized == "zmq":
        return ZmqRealtimePublisher(
            endpoint=kwargs.get("endpoint", "tcp://127.0.0.1:5556"),
            topic=kwargs.get("topic", "gaze.live"),
        )
    raise ValueError(f"Unknown realtime publisher backend: {backend}")
