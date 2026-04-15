"""Unit tests for Unreal realtime acceptance log parsing/evaluation."""

from __future__ import annotations

from src.ferret_gaze.realtime.unreal_e2e_acceptance import (
	UnrealAcceptanceMetrics,
	UnrealAcceptanceThresholds,
	evaluate_unreal_acceptance,
	parse_unreal_log_line,
)


def test_parse_subscriber_and_render_lines_updates_metrics() -> None:
	m = UnrealAcceptanceMetrics()
	parse_unreal_log_line(
		"GazeSubscriber connected | connected=1 reconnect_attempts=1 receive_errors=0 dropped=2 last_seq=41",
		m,
	)
	parse_unreal_log_line(
		"GazeRender stats | applied=60 stale_drop=3 conf_drop=1 age_ms[p50=12.2 p95=30.8 last=20.1]",
		m,
	)
	assert m.connected_events == 1
	assert m.max_last_seq == 41
	assert m.max_receive_errors == 0
	assert m.max_dropped == 2
	assert m.render_stats_events == 1
	assert m.max_applied == 60
	assert m.max_stale_drop == 3
	assert m.max_conf_drop == 1
	assert m.max_age_p95_ms == 30.8


def test_evaluate_unreal_acceptance_reports_failures() -> None:
	m = UnrealAcceptanceMetrics(
		connected_events=0,
		max_last_seq=5,
		render_stats_events=0,
		max_applied=4,
		max_receive_errors=2,
		max_stale_drop=999,
		max_conf_drop=999,
		max_age_p95_ms=1500.0,
	)
	t = UnrealAcceptanceThresholds(
		min_connected_events=1,
		min_last_seq=30,
		min_render_stats_events=1,
		min_applied_packets=20,
		max_receive_errors=0,
		max_stale_drop=20,
		max_conf_drop=20,
		max_age_p95_ms=500.0,
	)
	passed, failures = evaluate_unreal_acceptance(m, t)
	assert not passed
	assert len(failures) >= 6


def test_evaluate_unreal_acceptance_passes_when_within_thresholds() -> None:
	m = UnrealAcceptanceMetrics(
		connected_events=1,
		max_last_seq=120,
		render_stats_events=2,
		max_applied=120,
		max_receive_errors=0,
		max_stale_drop=4,
		max_conf_drop=2,
		max_age_p95_ms=55.0,
	)
	t = UnrealAcceptanceThresholds(
		min_connected_events=1,
		min_last_seq=30,
		min_render_stats_events=1,
		min_applied_packets=30,
		max_receive_errors=0,
		max_stale_drop=200,
		max_conf_drop=200,
		max_age_p95_ms=1000.0,
	)
	passed, failures = evaluate_unreal_acceptance(m, t)
	assert passed
	assert failures == []

