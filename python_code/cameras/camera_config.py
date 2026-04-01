"""
Camera hardware configuration for the Scholl Lab Basler rig.

Contains:
- ImageShape: width/height container.
- SERIAL_TO_IMAGE_SHAPE: default full-resolution shape per camera serial number.
- SERIAL_TO_EXPOSURE_GAIN: default exposure (µs) and gain per camera serial number.
- apply_camera_settings(): apply exposure and gain to a single camera.
- configure_all_cameras(): apply per-serial defaults to the whole array.
"""

from dataclasses import dataclass

import pypylon.pylon as pylon


@dataclass
class ImageShape:
    width: int
    height: int


# Default full-resolution shapes per Basler serial number.
SERIAL_TO_IMAGE_SHAPE: dict[str, ImageShape] = {
    "24908831": ImageShape(width=2048, height=2048),
    "24908832": ImageShape(width=2048, height=2048),
    "25000609": ImageShape(width=2048, height=2048),
    "25006505": ImageShape(width=2048, height=2048),
    "40520488": ImageShape(width=1920, height=1200),
    "24676894": ImageShape(width=1280, height=1024),
    "24678651": ImageShape(width=1280, height=1024),
}

# Default (exposure_us, gain) per serial number.
# These are used by configure_all_cameras() when no overrides are provided.
SERIAL_TO_EXPOSURE_GAIN: dict[str, tuple[int, float]] = {
    "24908831": (5000, 1.0),
    "24908832": (5000, 0.0),
    "25000609": (5000, 0.0),
    "25006505": (5000, 0.5),
    "40520488": (4500, 0.0),
    "24676894": (4500, 0.0),
    "24678651": (4500, 0.0),
}

# Serials that should not receive binning (fixed-resolution cameras).
NO_BINNING_SERIALS: frozenset[str] = frozenset({"40520488", "24676894"})


def get_image_shape(serial: str) -> ImageShape:
    """Return the default ImageShape for a given serial number."""
    if serial not in SERIAL_TO_IMAGE_SHAPE:
        raise ValueError(
            f"Serial number '{serial}' not found in SERIAL_TO_IMAGE_SHAPE. "
            f"Known serials: {sorted(SERIAL_TO_IMAGE_SHAPE)}"
        )
    return SERIAL_TO_IMAGE_SHAPE[serial]


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

    Settings are taken from SERIAL_TO_EXPOSURE_GAIN unless overridden.

    Args:
        camera_array: Open InstantCameraArray.
        devices: Device list matching camera_array order (used to resolve serials).
        overrides: Optional dict mapping serial → (exposure_us, gain) to override defaults.
    """
    merged = {**SERIAL_TO_EXPOSURE_GAIN, **(overrides or {})}

    for index, camera in enumerate(camera_array):
        serial = devices[index].GetSerialNumber()
        if serial not in merged:
            raise ValueError(
                f"Serial '{serial}' has no exposure/gain config. "
                f"Add it to SERIAL_TO_EXPOSURE_GAIN or pass overrides."
            )
        exposure, gain = merged[serial]
        apply_camera_settings(camera, exposure_time=exposure, gain=gain)
