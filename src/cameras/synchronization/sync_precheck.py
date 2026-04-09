"""
Inter-camera sync checks before streaming realtime packets (e.g. to Unreal).

Uses :class:`BaslerFrameSet` from the combiner: validate skew against a budget
stricter than the combiner tolerance if desired.
"""

from __future__ import annotations

from src.cameras.synchronization.realtime_sync import BaslerFrame, BaslerFrameSet


def max_inter_camera_skew_ns(frame_set: BaslerFrameSet) -> int:
	"""
	Maximum |capture_utc_ns - anchor_utc_ns| across cameras in the set.

	The combiner already enforces per-camera distance to the anchor; this
	re-exports the worst case for logging or tighter pre-publish gates.
	"""
	anchor = frame_set.anchor_utc_ns
	return max(abs(f.capture_utc_ns - anchor) for f in frame_set.frames_by_camera.values())


def validate_frame_set_before_unreal_publish(frame_set: BaslerFrameSet, *, max_skew_ns: int) -> None:
	"""
	Raise ValueError if multi-camera timestamp spread exceeds the publish budget.

	Call after :meth:`BaslerFrameSetCombiner.ingest` returns a set and before
	building/publishing :class:`~src.ferret_gaze.realtime.gaze_packet.RealtimeGazePacket`.
	"""
	if max_skew_ns < 0:
		raise ValueError("max_skew_ns must be non-negative")
	skew = max_inter_camera_skew_ns(frame_set)
	if skew > max_skew_ns:
		raise ValueError(
			f"inter-camera skew {skew} ns exceeds publish budget max_skew_ns={max_skew_ns} "
			f"(anchor_utc_ns={frame_set.anchor_utc_ns})"
		)


def synthetic_two_camera_ingest_sequence(
	*,
	num_frame_sets: int,
	base_utc_ns: int = 1_701_000_000_000_000_000,
	frame_period_ns: int = 16_666_666,
	inter_camera_offset_ns: int = 120_000,
) -> list[BaslerFrame]:
	"""
	Build ordered frames for :class:`BaslerFrameSetCombiner` tests and demos.

	Emits ``num_frame_sets`` pairs: camera 0 at ``t``, camera 1 at ``t + offset``.
	Ingest in list order; the combiner yields a set after each pair when the
	offset is within ``tolerance_ns``.
	"""
	if num_frame_sets <= 0:
		raise ValueError("num_frame_sets must be positive")
	frames: list[BaslerFrame] = []
	for i in range(num_frame_sets):
		t0 = base_utc_ns + i * frame_period_ns
		t1 = t0 + inter_camera_offset_ns
		frames.append(BaslerFrame(camera_id=0, frame_index=i, capture_utc_ns=t0))
		frames.append(BaslerFrame(camera_id=1, frame_index=i, capture_utc_ns=t1))
	return frames
