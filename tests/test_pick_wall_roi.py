"""Tests for wall ROI frame loading (no GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.arena_render.pick_wall_roi import _parse_roi_string, _read_frame


def test_read_frame_missing_file(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError):
		_read_frame(tmp_path / "missing.mp4", 0)


def test_read_frame_empty_file(tmp_path: Path) -> None:
	empty = tmp_path / "empty.mp4"
	empty.write_bytes(b"")
	with pytest.raises(RuntimeError, match="failed to open|Could not read"):
		_read_frame(empty, 0)


def test_parse_roi_string() -> None:
	assert _parse_roi_string("10,20,300,400") == (10, 20, 300, 400)


def test_parse_roi_string_rejects_bad_input() -> None:
	with pytest.raises(ValueError):
		_parse_roi_string("1,2,3")
