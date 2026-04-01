"""
Logging configuration for the cameras module.

Thin wrapper around the shared pipeline logger that sets a camera-specific
default log directory. All other modules should import get_logger from
python_code.utilities.logging_config directly.
"""

from pathlib import Path

from python_code.utilities.logging_config import get_logger

# Camera-specific log directory (lab machine path).
_CAMERA_LOG_DIR = Path("/home/scholl-lab/recordings/basler_logs")


def get_camera_logger(name: str) -> "logging.Logger":
    """
    Return a named logger configured for the cameras module.

    Uses the shared pipeline root logger with the camera-specific log
    directory on first call. Arguments beyond name are no longer needed
    as the shared config handles level and format uniformly.

    Args:
        name: Logger name, typically __name__ of the calling module.
    """
    import logging
    return get_logger(name, log_dir=_CAMERA_LOG_DIR)
