"""Soft-limit workspace helpers for gantry prey motion."""

from __future__ import annotations

import math

from src.hunt.config import WorkspaceMm


def clip_xy(x: float, y: float, ws: WorkspaceMm) -> tuple[float, float]:
	"""Clamp a point into the soft-limit rectangle."""
	return (
		min(max(x, ws.x_min), ws.x_max),
		min(max(y, ws.y_min), ws.y_max),
	)


def point_in_workspace(x: float, y: float, ws: WorkspaceMm, eps: float = 1e-6) -> bool:
	return (
		ws.x_min - eps <= x <= ws.x_max + eps
		and ws.y_min - eps <= y <= ws.y_max + eps
	)


def corner_points(ws: WorkspaceMm, margin_mm: float) -> list[tuple[float, float]]:
	"""Four corners inset by margin so moves do not slam soft limits."""
	x0, x1 = ws.x_min + margin_mm, ws.x_max - margin_mm
	y0, y1 = ws.y_min + margin_mm, ws.y_max - margin_mm
	if x0 >= x1 or y0 >= y1:
		raise ValueError("corner_margin_mm leaves empty workspace")
	return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def bearing_delta_mm(distance_mm: float, bearing_deg: float) -> tuple[float, float]:
	rad = math.radians(bearing_deg)
	return distance_mm * math.cos(rad), distance_mm * math.sin(rad)


def clip_move(
	x: float,
	y: float,
	dx: float,
	dy: float,
	ws: WorkspaceMm,
) -> tuple[float, float] | None:
	"""End point of a relative move clipped to workspace; None if zero length."""
	ex, ey = clip_xy(x + dx, y + dy, ws)
	if math.hypot(ex - x, ey - y) < 1e-3:
		return None
	return ex, ey


def grid_points(ws: WorkspaceMm, spacing_mm: float) -> list[tuple[float, float]]:
	"""Inclusive grid covering the soft-limit rectangle."""
	if spacing_mm <= 0:
		raise ValueError("coverage_grid_mm must be > 0")
	pts: list[tuple[float, float]] = []
	x = ws.x_min
	while x <= ws.x_max + 1e-9:
		y = ws.y_min
		while y <= ws.y_max + 1e-9:
			pts.append((min(x, ws.x_max), min(y, ws.y_max)))
			y += spacing_mm
		x += spacing_mm
	# Ensure far corner included when spacing does not land exactly.
	far = (ws.x_max, ws.y_max)
	if far not in pts:
		pts.append(far)
	return pts
