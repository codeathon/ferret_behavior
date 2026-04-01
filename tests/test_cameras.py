"""
Unit tests for the refactored python_code/cameras package.

Covers the five focused sub-modules introduced during the camera_restructure:
    camera_config   — CameraProfile, CAMERAS, derived constants, helper functions
    logging_config  — get_camera_logger: file handler, console handler, fallback
    timestamp_utils — trim_timestamp_zeros (pure), save_timestamps (I/O)
    video_writers   — VideoWriterManager guard conditions and state management
    grab_loops      — GrabLoopRunner pure logic (seconds→frames, statistics return)

Hardware-dependent code (actual pypylon cameras, ffmpeg subprocesses) is replaced
with lightweight mocks so the suite runs on any machine without Basler hardware.
pypylon.pylon itself is patched at sys.modules level before any camera import so
the import-time `import pypylon.pylon as pylon` line does not fail.
"""

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Patch pypylon before any cameras-module import
# ---------------------------------------------------------------------------

def _make_pylon_stub() -> types.ModuleType:
    """Return a minimal stub of pypylon.pylon sufficient for import."""
    pylon = types.ModuleType("pypylon.pylon")
    pylon.TlFactory = MagicMock()
    pylon.InstantCamera = MagicMock()
    pylon.InstantCameraArray = MagicMock()
    pylon.GrabStrategy_LatestImageOnly = 0
    return pylon


_pylon_stub = _make_pylon_stub()
sys.modules.setdefault("pypylon", types.ModuleType("pypylon"))
sys.modules.setdefault("pypylon.pylon", _pylon_stub)

# Now safe to import camera modules
from python_code.cameras.camera_config import (  # noqa: E402
    CAMERAS,
    CameraProfile,
    ImageShape,
    KNOWN_SERIALS,
    NO_BINNING_SERIALS,
    SERIAL_TO_EXPOSURE_GAIN,
    SERIAL_TO_IMAGE_SHAPE,
    apply_camera_settings,
    configure_all_cameras,
    get_camera_profile,
    get_image_shape,
)
from python_code.cameras.logging_config import get_camera_logger  # noqa: E402
from python_code.cameras.timestamp_utils import (  # noqa: E402
    save_timestamps,
    trim_timestamp_zeros,
)


# ===========================================================================
# camera_config
# ===========================================================================

class TestImageShape:
    def test_fields_accessible(self):
        s = ImageShape(width=1920, height=1080)
        assert s.width == 1920
        assert s.height == 1080

    def test_frozen(self):
        s = ImageShape(width=100, height=200)
        with pytest.raises((TypeError, AttributeError)):
            s.width = 999  # type: ignore[misc]


class TestCameraProfile:
    def test_fields_accessible(self):
        profile = CameraProfile(
            serial="99999999",
            image_shape=ImageShape(2048, 2048),
            default_exposure_us=5000,
            default_gain=1.0,
            binning_allowed=True,
        )
        assert profile.serial == "99999999"
        assert profile.image_shape.width == 2048
        assert profile.default_exposure_us == 5000
        assert profile.default_gain == 1.0
        assert profile.binning_allowed is True

    def test_frozen(self):
        profile = CameraProfile("0", ImageShape(1, 1), 100, 0.0, True)
        with pytest.raises((TypeError, AttributeError)):
            profile.serial = "new"  # type: ignore[misc]


class TestCAMERAS:
    def test_all_known_serials_present(self):
        expected = {
            "24908831", "24908832", "25000609", "25006505",
            "40520488", "24676894", "24678651",
        }
        assert set(CAMERAS.keys()) == expected

    def test_all_values_are_camera_profiles(self):
        for serial, profile in CAMERAS.items():
            assert isinstance(profile, CameraProfile), f"{serial} is not a CameraProfile"

    def test_serial_key_matches_profile_serial(self):
        for key, profile in CAMERAS.items():
            assert key == profile.serial, f"Key {key!r} != profile.serial {profile.serial!r}"

    def test_image_shapes_are_positive(self):
        for serial, profile in CAMERAS.items():
            assert profile.image_shape.width > 0, f"{serial}: width <= 0"
            assert profile.image_shape.height > 0, f"{serial}: height <= 0"

    def test_exposure_is_positive(self):
        for serial, profile in CAMERAS.items():
            assert profile.default_exposure_us > 0, f"{serial}: exposure <= 0"

    def test_gain_is_non_negative(self):
        for serial, profile in CAMERAS.items():
            assert profile.default_gain >= 0.0, f"{serial}: gain < 0"


class TestDerivedConstants:
    def test_known_serials_matches_cameras_keys(self):
        assert KNOWN_SERIALS == frozenset(CAMERAS.keys())

    def test_serial_to_image_shape_derived(self):
        for serial, profile in CAMERAS.items():
            assert SERIAL_TO_IMAGE_SHAPE[serial] == profile.image_shape

    def test_serial_to_exposure_gain_derived(self):
        for serial, profile in CAMERAS.items():
            exp, gain = SERIAL_TO_EXPOSURE_GAIN[serial]
            assert exp == profile.default_exposure_us
            assert gain == profile.default_gain

    def test_no_binning_serials_derived(self):
        expected = frozenset(s for s, p in CAMERAS.items() if not p.binning_allowed)
        assert NO_BINNING_SERIALS == expected

    def test_no_binning_serials_are_subset_of_known(self):
        assert NO_BINNING_SERIALS <= KNOWN_SERIALS

    def test_no_binning_does_not_include_binning_allowed(self):
        for serial in NO_BINNING_SERIALS:
            assert not CAMERAS[serial].binning_allowed


class TestGetCameraProfile:
    def test_returns_correct_profile(self):
        profile = get_camera_profile("24908831")
        assert isinstance(profile, CameraProfile)
        assert profile.serial == "24908831"

    def test_raises_on_unknown_serial(self):
        with pytest.raises(ValueError, match="99999999"):
            get_camera_profile("99999999")

    def test_error_message_lists_known_serials(self):
        with pytest.raises(ValueError, match="24908831"):
            get_camera_profile("bad_serial")


class TestGetImageShape:
    def test_known_serial_returns_image_shape(self):
        shape = get_image_shape("24908831")
        assert isinstance(shape, ImageShape)
        assert shape.width == 2048
        assert shape.height == 2048

    def test_unknown_serial_raises(self):
        with pytest.raises(ValueError):
            get_image_shape("00000000")

    @pytest.mark.parametrize("serial", list(CAMERAS.keys()))
    def test_all_serials_resolvable(self, serial):
        shape = get_image_shape(serial)
        assert shape == CAMERAS[serial].image_shape


class TestApplyCameraSettings:
    def test_sets_exposure_and_gain(self):
        mock_camera = MagicMock()
        apply_camera_settings(mock_camera, exposure_time=5000, gain=1.5)
        assert mock_camera.ExposureTime.Value == 5000
        assert mock_camera.Gain.Value == 1.5

    def test_zero_gain_accepted(self):
        mock_camera = MagicMock()
        apply_camera_settings(mock_camera, exposure_time=1000, gain=0.0)
        assert mock_camera.Gain.Value == 0.0


class TestConfigureAllCameras:
    def _make_camera(self, serial: str) -> MagicMock:
        cam = MagicMock()
        cam.DeviceInfo.GetSerialNumber.return_value = serial
        return cam

    def _make_device(self, serial: str) -> MagicMock:
        dev = MagicMock()
        dev.GetSerialNumber.return_value = serial
        return dev

    def _make_array(self, serials: list[str]):
        cameras = [self._make_camera(s) for s in serials]
        array = MagicMock()
        array.__iter__ = MagicMock(return_value=iter(cameras))
        return array, cameras

    def test_applies_defaults_from_cameras(self):
        serials = ["24908831", "24908832"]
        array, cam_mocks = self._make_array(serials)
        devices = [self._make_device(s) for s in serials]

        configure_all_cameras(array, devices)

        for i, serial in enumerate(serials):
            exp, gain = SERIAL_TO_EXPOSURE_GAIN[serial]
            assert cam_mocks[i].ExposureTime.Value == exp
            assert cam_mocks[i].Gain.Value == gain

    def test_overrides_take_precedence(self):
        serial = "24908831"
        array, cam_mocks = self._make_array([serial])
        devices = [self._make_device(serial)]

        configure_all_cameras(array, devices, overrides={serial: (9999, 9.9)})

        assert cam_mocks[0].ExposureTime.Value == 9999
        assert cam_mocks[0].Gain.Value == 9.9

    def test_unknown_serial_raises(self):
        array, _ = self._make_array(["unknown_serial"])
        devices = [self._make_device("unknown_serial")]
        with pytest.raises(ValueError, match="unknown_serial"):
            configure_all_cameras(array, devices)


# ===========================================================================
# logging_config
# ===========================================================================

class TestGetCameraLogger:
    def test_returns_logger(self):
        logger = get_camera_logger("test.cameras.a")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        logger = get_camera_logger("test.cameras.b")
        assert logger.name == "test.cameras.b"

    def test_console_handler_present(self):
        logger = get_camera_logger("test.cameras.c")
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) >= 1

    def test_file_handler_created_when_dir_exists(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        logger = get_camera_logger("test.cameras.file", log_dir=log_dir)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_fallback_to_console_when_dir_not_writable(self, tmp_path):
        nonexistent = tmp_path / "no" / "such" / "path"
        # Make parent read-only so mkdir fails
        tmp_path.chmod(0o444)
        try:
            logger = get_camera_logger("test.cameras.fallback", log_dir=nonexistent)
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 0
        finally:
            tmp_path.chmod(0o755)

    def test_idempotent_on_second_call(self):
        name = "test.cameras.idempotent"
        logger1 = get_camera_logger(name)
        n_handlers = len(logger1.handlers)
        logger2 = get_camera_logger(name)
        assert logger1 is logger2
        assert len(logger2.handlers) == n_handlers


# ===========================================================================
# timestamp_utils
# ===========================================================================

class TestTrimTimestampZeros:
    def test_trims_trailing_zeros(self):
        ts = np.array([[1, 2, 3, 0, 0],
                       [4, 5, 6, 0, 0]], dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, ts[:, :3])

    def test_no_trailing_zeros(self):
        ts = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        assert result.shape == (2, 3)

    def test_all_zeros_returns_empty(self):
        ts = np.zeros((3, 10), dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        assert result.shape[1] == 0

    def test_single_frame(self):
        ts = np.array([[1], [2]], dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        assert result.shape == (2, 1)

    def test_preserves_nonzero_values(self):
        ts = np.array([[10, 20, 0],
                       [30, 40, 0]], dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        np.testing.assert_array_equal(result, np.array([[10, 20], [30, 40]]))

    def test_single_camera(self):
        ts = np.array([[5, 10, 15, 0, 0]], dtype=np.int64)
        result = trim_timestamp_zeros(ts)
        assert result.shape == (1, 3)


class TestSaveTimestamps:
    def _make_mapping(self, camera_ids: list[int]):
        from python_code.cameras.diagnostics.timestamp_mapping import TimestampMapping
        return TimestampMapping(camera_timestamps={i: i * 1000 for i in camera_ids})

    def test_saves_npy_file(self, tmp_path):
        ts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        start = self._make_mapping([0, 1])
        end = self._make_mapping([0, 1])
        save_timestamps(tmp_path, ts, start, end)
        assert (tmp_path / "timestamps.npy").exists()

    def test_npy_content_is_trimmed(self, tmp_path):
        ts = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.int64)
        start = self._make_mapping([0, 1])
        end = self._make_mapping([0, 1])
        save_timestamps(tmp_path, ts, start, end)
        loaded = np.load(tmp_path / "timestamps.npy")
        assert loaded.shape == (2, 2)

    def test_saves_json_mapping(self, tmp_path):
        ts = np.array([[1, 2], [3, 4]], dtype=np.int64)
        start = self._make_mapping([0, 1])
        end = self._make_mapping([0, 1])
        save_timestamps(tmp_path, ts, start, end)
        mapping_path = tmp_path / "timestamp_mapping.json"
        assert mapping_path.exists()
        data = json.loads(mapping_path.read_text())
        assert "starting_mapping" in data
        assert "ending_mapping" in data

    def test_json_contains_camera_timestamps(self, tmp_path):
        ts = np.array([[1], [2]], dtype=np.int64)
        start = self._make_mapping([0, 1])
        end = self._make_mapping([0, 1])
        save_timestamps(tmp_path, ts, start, end)
        data = json.loads((tmp_path / "timestamp_mapping.json").read_text())
        assert "camera_timestamps" in data["starting_mapping"]

    def test_raises_if_json_already_exists(self, tmp_path):
        """save_timestamps opens with mode='x', so duplicate calls should fail."""
        ts = np.array([[1], [2]], dtype=np.int64)
        start = self._make_mapping([0, 1])
        end = self._make_mapping([0, 1])
        save_timestamps(tmp_path, ts, start, end)
        with pytest.raises(FileExistsError):
            save_timestamps(tmp_path, ts, start, end)


# ===========================================================================
# video_writers — guard conditions (no real ffmpeg/cv2 needed)
# ===========================================================================

# Patch cv2 and ffmpeg at import time so VideoWriterManager can be imported
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("ffmpeg", MagicMock())

from python_code.cameras.video_writers import VideoWriterManager  # noqa: E402


class TestVideoWriterManagerGuards:
    def _make_manager(self) -> VideoWriterManager:
        array = MagicMock()
        array.__iter__ = MagicMock(return_value=iter([]))
        shapes: dict = {}
        return VideoWriterManager(
            camera_array=array,
            image_shapes=shapes,
            fps=30.0,
            output_path=Path("/tmp/fake"),
        )

    def test_write_opencv_before_create_raises(self):
        mgr = self._make_manager()
        with pytest.raises(RuntimeError, match="not initialised"):
            mgr.write_frame_opencv(np.zeros((10, 10), dtype=np.uint8), cam_id=0, frame_number=1)

    def test_write_ffmpeg_before_create_raises(self):
        mgr = self._make_manager()
        with pytest.raises(RuntimeError, match="not initialised"):
            mgr.write_frame_ffmpeg(np.zeros((10, 10), dtype=np.uint8), cam_id=0)

    def test_release_opencv_when_none_is_safe(self):
        mgr = self._make_manager()
        mgr.release_opencv()  # should not raise

    def test_release_ffmpeg_when_none_is_safe(self):
        mgr = self._make_manager()
        mgr.release_ffmpeg()  # should not raise

    def test_update_fps_stores_new_value(self):
        mgr = self._make_manager()
        mgr.update_fps(60.0)
        assert mgr._fps == 60.0

    def test_update_image_shapes_stores_new_shapes(self):
        mgr = self._make_manager()
        new_shapes = {0: ImageShape(1280, 720)}
        mgr.update_image_shapes(new_shapes)
        assert mgr._image_shapes == new_shapes


# ===========================================================================
# grab_loops — pure logic
# ===========================================================================

from python_code.cameras.grab_loops import GrabLoopRunner  # noqa: E402


class TestGrabLoopRunnerPureLogic:
    def _make_runner(self, fps: float = 30.0) -> GrabLoopRunner:
        camera_array = MagicMock()
        camera_array.__iter__ = MagicMock(return_value=iter([]))
        writer = MagicMock()
        return GrabLoopRunner(
            camera_array=camera_array,
            n_cameras=2,
            fps=fps,
            writer=writer,
            output_path=Path("/tmp/fake"),
        )

    def test_grab_n_seconds_converts_correctly(self):
        runner = self._make_runner(fps=30.0)
        with patch.object(runner, "_run") as mock_run:
            runner.grab_n_seconds(2.0)
            args, kwargs = mock_run.call_args
            # max_frames should be int(2.0 * 30) = 60
            assert kwargs.get("max_frames", args[1] if len(args) > 1 else None) == 60

    def test_grab_n_seconds_at_90fps(self):
        runner = self._make_runner(fps=90.0)
        with patch.object(runner, "_run") as mock_run:
            runner.grab_n_seconds(10.0)
            _, kwargs = mock_run.call_args
            assert kwargs.get("max_frames") == 900

    def test_pylon_statistics_returns_true_when_no_failures(self):
        runner = self._make_runner()
        cam1, cam2 = MagicMock(), MagicMock()
        cam1.StreamGrabber.Statistic_Total_Buffer_Count.GetValue.return_value = 100
        cam1.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue.return_value = 0
        cam1.GetCameraContext.return_value = 0
        cam2.StreamGrabber.Statistic_Total_Buffer_Count.GetValue.return_value = 100
        cam2.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue.return_value = 0
        cam2.GetCameraContext.return_value = 1
        runner._camera_array.__iter__ = MagicMock(return_value=iter([cam1, cam2]))
        assert runner.pylon_internal_statistics() is True

    def test_pylon_statistics_returns_false_when_frames_dropped(self):
        runner = self._make_runner()
        cam = MagicMock()
        cam.StreamGrabber.Statistic_Total_Buffer_Count.GetValue.return_value = 100
        cam.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue.return_value = 5
        cam.GetCameraContext.return_value = 0
        runner._camera_array.__iter__ = MagicMock(return_value=iter([cam]))
        assert runner.pylon_internal_statistics() is False

    def test_grab_n_frames_condition_true_at_target(self):
        """The condition lambda should return True exactly when min(counts) >= n."""
        runner = self._make_runner()
        with patch.object(runner, "_run") as mock_run:
            runner.grab_n_frames(50)
            _, kwargs = mock_run.call_args
            condition = kwargs["condition"]
            assert condition([50, 50]) is True
            assert condition([49, 50]) is False
            assert condition([51, 51]) is True
