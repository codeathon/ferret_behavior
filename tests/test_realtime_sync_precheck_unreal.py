"""
Dummy multi-camera data + pre-Unreal sync validation.

Ensures combined Basler frame-sets are within a publish skew budget before a
realtime packet would be sent to Unreal.
"""

import pytest

from src.cameras.synchronization.realtime_sync import BaslerFrame, BaslerFrameSet, BaslerFrameSetCombiner
from src.cameras.synchronization.sync_precheck import (
	max_inter_camera_skew_ns,
	synthetic_two_camera_ingest_sequence,
	validate_frame_set_before_unreal_publish,
)


def test_synthetic_stream_passes_combiner_and_precheck() -> None:
	"""Aligned dummy captures: combiner emits sets; precheck accepts skew under budget."""
	tolerance_ns = 2_000_000
	frames = synthetic_two_camera_ingest_sequence(
		num_frame_sets=5,
		inter_camera_offset_ns=150_000,
	)
	combiner = BaslerFrameSetCombiner(camera_ids=[0, 1], ring_size=16, tolerance_ns=tolerance_ns)
	emitted: list = []
	for f in frames:
		fs = combiner.ingest(f)
		if fs is not None:
			emitted.append(fs)
	assert len(emitted) == 5
	for fs in emitted:
		skew = max_inter_camera_skew_ns(fs)
		assert skew == 150_000
		# Tighter than combiner tolerance but still above measured skew.
		validate_frame_set_before_unreal_publish(fs, max_skew_ns=200_000)


def test_precheck_rejects_manual_frame_set_over_budget() -> None:
	"""If skew exceeds publish budget, gate raises before hypothetical Unreal send."""
	fs = BaslerFrameSet(
		anchor_utc_ns=1_000_000_000,
		frames_by_camera={
			0: BaslerFrame(camera_id=0, frame_index=0, capture_utc_ns=1_000_000_000),
			1: BaslerFrame(camera_id=1, frame_index=0, capture_utc_ns=1_000_000_000 + 500_000),
		},
	)
	validate_frame_set_before_unreal_publish(fs, max_skew_ns=600_000)
	with pytest.raises(ValueError, match="inter-camera skew"):
		validate_frame_set_before_unreal_publish(fs, max_skew_ns=400_000)


def test_combiner_drops_grossly_misaligned_dummy_stream() -> None:
	"""Dummy offset beyond combiner tolerance: no frame-set to publish."""
	frames = synthetic_two_camera_ingest_sequence(
		num_frame_sets=3,
		inter_camera_offset_ns=5_000_000,
	)
	combiner = BaslerFrameSetCombiner(camera_ids=[0, 1], ring_size=16, tolerance_ns=2_000_000)
	for f in frames:
		assert combiner.ingest(f) is None
