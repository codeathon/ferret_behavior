"""
Wire serialization for RealtimeGazePacket (ZMQ payload for Unreal).

Keys and value shapes match FerretGazeLive ParsePayloadMsgpack expectations.
"""

from __future__ import annotations

from src.ferret_gaze.realtime.gaze_packet import RealtimeGazePacket


def gaze_packet_to_wire_dict(packet: RealtimeGazePacket) -> dict[str, object]:
    """
    Build a JSON/msgpack-friendly dict with stable keys for the UE subscriber.

    Omits optional fields when None so Unreal leaves those USTRUCT fields at
    defaults (zero / identity) unless the parser requires presence.
    """
    out: dict[str, object] = {
        "seq": packet.seq,
        "capture_utc_ns": packet.capture_utc_ns,
        "skull_position_xyz": tuple(packet.skull_position_xyz),
        "skull_quaternion_wxyz": tuple(packet.skull_quaternion_wxyz),
        "left_eye_origin_xyz": tuple(packet.left_eye_origin_xyz),
        "left_gaze_direction_xyz": tuple(packet.left_gaze_direction_xyz),
        "right_eye_origin_xyz": tuple(packet.right_eye_origin_xyz),
        "right_gaze_direction_xyz": tuple(packet.right_gaze_direction_xyz),
    }
    if packet.process_start_ns is not None:
        out["process_start_ns"] = packet.process_start_ns
    if packet.publish_utc_ns is not None:
        out["publish_utc_ns"] = packet.publish_utc_ns
    if packet.confidence is not None:
        out["confidence"] = packet.confidence
    return out
