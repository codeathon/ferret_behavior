"""
Realtime gaze packet schema for Python -> Unreal streaming.

The realtime pipeline publishes one packet per logical frame. This model keeps
the transport schema explicit and validates basic shape constraints.
"""

from pydantic import BaseModel, Field, field_validator


class RealtimeGazePacket(BaseModel):
    """Single-frame gaze payload in world coordinates."""

    seq: int = Field(ge=0)
    capture_utc_ns: int = Field(ge=0)
    process_start_ns: int | None = Field(default=None, ge=0)
    publish_utc_ns: int | None = Field(default=None, ge=0)

    # Skull pose in world coordinates (project convention quaternion is wxyz).
    skull_position_xyz: tuple[float, float, float]
    skull_quaternion_wxyz: tuple[float, float, float, float]

    # Eye origins and gaze vectors in world coordinates.
    left_eye_origin_xyz: tuple[float, float, float]
    left_gaze_direction_xyz: tuple[float, float, float]
    right_eye_origin_xyz: tuple[float, float, float]
    right_gaze_direction_xyz: tuple[float, float, float]

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "skull_position_xyz",
        "left_eye_origin_xyz",
        "left_gaze_direction_xyz",
        "right_eye_origin_xyz",
        "right_gaze_direction_xyz",
        mode="after",
    )
    @classmethod
    def _validate_vec3(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Ensure vector fields have exactly 3 numeric values."""
        if len(value) != 3:
            raise ValueError("Expected a 3D tuple")
        return value

    @field_validator("skull_quaternion_wxyz", mode="after")
    @classmethod
    def _validate_quat(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Ensure quaternion has exactly 4 numeric values."""
        if len(value) != 4:
            raise ValueError("Expected a 4D quaternion tuple in wxyz order")
        return value
