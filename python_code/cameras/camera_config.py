"""
Camera hardware configuration for the Scholl Lab Basler rig.

All per-camera properties are defined once in the CAMERAS dict as CameraProfile
entries. Every other constant and helper function is derived from that single
source of truth, so adding, removing, or reconfiguring a camera only requires
editing one row.

Exports:
    CameraProfile          — frozen dataclass holding all per-camera properties.
    CAMERAS                — dict[serial_str, CameraProfile] — the single source of truth.
    KNOWN_SERIALS          — frozenset of all registered serial number strings.
    SERIAL_TO_IMAGE_SHAPE  — derived dict[serial, ImageShape] (for backwards compat).
    SERIAL_TO_EXPOSURE_GAIN— derived dict[serial, (exposure_us, gain)].
    NO_BINNING_SERIALS     — derived frozenset of serials that must not be binned.
    ImageShape             — simple width/height container.
    get_image_shape()      — look up ImageShape by serial, raises on unknown serial.
    apply_camera_settings()— apply exposure + gain to a single InstantCamera.
    configure_all_cameras()— apply defaults (with optional overrides) to whole array.
"""

from dataclasses import dataclass

import pypylon.pylon as pylon


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageShape:
    width: int
    height: int


@dataclass(frozen=True)
class CameraProfile:
    """All hardware properties for one Basler camera."""
    serial: str
    image_shape: ImageShape
    default_exposure_us: int
    default_gain: float
    binning_allowed: bool


# ---------------------------------------------------------------------------
# Single source of truth — add / remove / edit cameras here only
# ---------------------------------------------------------------------------

CAMERAS: dict[str, CameraProfile] = {
    "24908831": CameraProfile("24908831", ImageShape(2048, 2048), default_exposure_us=5000, default_gain=1.0, binning_allowed=True),
    "24908832": CameraProfile("24908832", ImageShape(2048, 2048), default_exposure_us=5000, default_gain=0.0, binning_allowed=True),
    "25000609": CameraProfile("25000609", ImageShape(2048, 2048), default_exposure_us=5000, default_gain=0.0, binning_allowed=True),
    "25006505": CameraProfile("25006505", ImageShape(2048, 2048), default_exposure_us=5000, default_gain=0.5, binning_allowed=True),
    "40520488": CameraProfile("40520488", ImageShape(1920, 1200), default_exposure_us=4500, default_gain=0.0, binning_allowed=False),
    "24676894": CameraProfile("24676894", ImageShape(1280, 1024), default_exposure_us=4500, default_gain=0.0, binning_allowed=False),
    "24678651": CameraProfile("24678651", ImageShape(1280, 1024), default_exposure_us=4500, default_gain=0.0, binning_allowed=True),
}


# ---------------------------------------------------------------------------
# Derived constants — do not edit; update CAMERAS above instead
# ---------------------------------------------------------------------------

KNOWN_SERIALS: frozenset[str] = frozenset(CAMERAS)

SERIAL_TO_IMAGE_SHAPE: dict[str, ImageShape] = {
    s: p.image_shape for s, p in CAMERAS.items()
}

SERIAL_TO_EXPOSURE_GAIN: dict[str, tuple[int, float]] = {
    s: (p.default_exposure_us, p.default_gain) for s, p in CAMERAS.items()
}

NO_BINNING_SERIALS: frozenset[str] = frozenset(
    s for s, p in CAMERAS.items() if not p.binning_allowed
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_camera_profile(serial: str) -> CameraProfile:
    """Return the CameraProfile for a given serial number."""
    if serial not in CAMERAS:
        raise ValueError(
            f"Serial '{serial}' is not registered in CAMERAS. "
            f"Known serials: {sorted(KNOWN_SERIALS)}"
        )
    return CAMERAS[serial]


def get_image_shape(serial: str) -> ImageShape:
    """Return the default ImageShape for a given serial number."""
    return get_camera_profile(serial).image_shape


def apply_camera_settings(
    camera: pylon.InstantCamera,
    exposure_time: int,
    gain: float,
) -> None:
    """
    Apply exposure time and gain to a single camera.

    Args:
        camera: An open InstantCamera instance.
        exposure_time: Exposure time in microseconds.
        gain: Gain value (camera-specific units).
    """
    camera.ExposureTime.Value = exposure_time
    camera.Gain.Value = gain


def configure_all_cameras(
    camera_array: pylon.InstantCameraArray,
    devices: list,
    overrides: dict[str, tuple[int, float]] | None = None,
) -> None:
    """
    Apply exposure and gain settings to every camera in the array.

    Defaults are taken from each camera's CameraProfile. Pass overrides to
    change specific cameras without touching the profile definitions.

    Args:
        camera_array: Open InstantCameraArray.
        devices: Device list matching camera_array index order.
        overrides: Optional dict mapping serial → (exposure_us, gain).
    """
    merged = {**SERIAL_TO_EXPOSURE_GAIN, **(overrides or {})}

    for index, camera in enumerate(camera_array):
        serial = devices[index].GetSerialNumber()
        if serial not in merged:
            raise ValueError(
                f"Serial '{serial}' has no exposure/gain config. "
                f"Add it to CAMERAS in camera_config.py or pass overrides."
            )
        exposure, gain = merged[serial]
        apply_camera_settings(camera, exposure_time=exposure, gain=gain)
