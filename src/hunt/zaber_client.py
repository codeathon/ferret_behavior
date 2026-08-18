"""Zaber gantry client — real ASCII device or fake for dry-run / CI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from src.hunt.config import HuntConfig, WorkspaceMm
from src.hunt.workspace import clip_xy, point_in_workspace


@dataclass
class PoseMm:
	x: float
	y: float


class Gantry(Protocol):
	def connect(self) -> None: ...
	def close(self) -> None: ...
	def home(self) -> None: ...
	def get_xy(self) -> PoseMm: ...
	def move_abs_mm(self, x: float, y: float, speed_mps: float) -> float: ...
	def stop(self) -> None: ...


class FakeGantry:
	"""In-memory stage for dry-run and unit tests (no serial)."""

	def __init__(self, workspace: WorkspaceMm, settle_tol_mm: float = 1.0):
		self._ws = workspace
		self._tol = settle_tol_mm
		self._pose = PoseMm(
			0.5 * (workspace.x_min + workspace.x_max),
			0.5 * (workspace.y_min + workspace.y_max),
		)
		self._connected = False

	def connect(self) -> None:
		self._connected = True

	def close(self) -> None:
		self._connected = False

	def home(self) -> None:
		self._require()
		self._pose = PoseMm(self._ws.x_min, self._ws.y_min)

	def get_xy(self) -> PoseMm:
		self._require()
		return PoseMm(self._pose.x, self._pose.y)

	def move_abs_mm(self, x: float, y: float, speed_mps: float) -> float:
		"""Move and return host-side duration seconds (simulated travel time)."""
		self._require()
		tx, ty = clip_xy(x, y, self._ws)
		if not point_in_workspace(tx, ty, self._ws):
			raise ValueError(f"target outside workspace: ({tx}, {ty})")
		dist_m = ((tx - self._pose.x) ** 2 + (ty - self._pose.y) ** 2) ** 0.5 / 1000.0
		speed = max(speed_mps, 1e-3)
		dt = dist_m / speed
		time.sleep(min(dt, 0.05))  # cap sleep in fake mode so CI stays fast
		self._pose = PoseMm(tx, ty)
		return dt

	def stop(self) -> None:
		self._require()

	def _require(self) -> None:
		if not self._connected:
			raise RuntimeError("FakeGantry not connected")


class ZaberGantry:
	"""Thin wrapper around zaber_motion ASCII for an XY (optional XXY lockstep) stage."""

	def __init__(self, cfg: HuntConfig):
		self._cfg = cfg
		self._conn = None
		self._device = None
		self._axis_x = None
		self._axis_y = None
		self._lockstep = None

	def connect(self) -> None:
		from zaber_motion.ascii import Connection

		self._conn = Connection.open_serial_port(self._cfg.port, self._cfg.baud_rate)
		self._conn.enable_alerts()
		devices = self._conn.detect_devices()
		if not devices:
			raise RuntimeError(f"No Zaber devices on {self._cfg.port}")
		# Prefer configured address; else first device.
		self._device = next(
			(d for d in devices if d.device_address == self._cfg.device_address),
			devices[0],
		)
		self._axis_x = self._device.get_axis(self._cfg.axis_x)
		self._axis_y = self._device.get_axis(self._cfg.axis_y)
		if self._cfg.axis_x2 > 0:
			# XXY: enable lockstep so dual-X stays yoked.
			self._lockstep = self._device.get_lockstep(self._cfg.lockstep_group)
			try:
				self._lockstep.enable(self._cfg.axis_x, self._cfg.axis_x2)
			except Exception as exc:
				# Already enabled is fine; surface other failures.
				if "already" not in str(exc).lower():
					raise

	def close(self) -> None:
		if self._conn is not None:
			self._conn.close()
		self._conn = None

	def home(self) -> None:
		self._require()
		if self._lockstep is not None:
			self._lockstep.home()
		else:
			self._axis_x.home()
		self._axis_y.home()

	def get_xy(self) -> PoseMm:
		self._require()
		from zaber_motion import Units

		x = self._axis_x.get_position(Units.LENGTH_MILLIMETRES)
		y = self._axis_y.get_position(Units.LENGTH_MILLIMETRES)
		return PoseMm(float(x), float(y))

	def move_abs_mm(self, x: float, y: float, speed_mps: float) -> float:
		self._require()
		from zaber_motion import Units

		tx, ty = clip_xy(x, y, self._cfg.workspace)
		speed_mm_s = max(speed_mps, 1e-3) * 1000.0
		t0 = time.perf_counter()
		# Prefer velocity kw on move_absolute when supported by the installed SDK.
		try:
			if self._lockstep is not None:
				self._lockstep.move_absolute(
					tx,
					Units.LENGTH_MILLIMETRES,
					velocity=speed_mm_s,
					velocity_unit=Units.VELOCITY_MILLIMETRES_PER_SECOND,
				)
			else:
				self._axis_x.move_absolute(
					tx,
					Units.LENGTH_MILLIMETRES,
					wait_until_idle=False,
					velocity=speed_mm_s,
					velocity_unit=Units.VELOCITY_MILLIMETRES_PER_SECOND,
				)
			self._axis_y.move_absolute(
				ty,
				Units.LENGTH_MILLIMETRES,
				wait_until_idle=False,
				velocity=speed_mm_s,
				velocity_unit=Units.VELOCITY_MILLIMETRES_PER_SECOND,
			)
		except TypeError:
			# Older bindings without velocity kwargs.
			if self._lockstep is not None:
				self._lockstep.move_absolute(tx, Units.LENGTH_MILLIMETRES)
			else:
				self._axis_x.move_absolute(tx, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
			self._axis_y.move_absolute(ty, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
		self._axis_x.wait_until_idle()
		self._axis_y.wait_until_idle()
		return time.perf_counter() - t0

	def stop(self) -> None:
		self._require()
		self._axis_x.stop()
		self._axis_y.stop()

	def _require(self) -> None:
		if self._device is None:
			raise RuntimeError("ZaberGantry not connected")


def open_gantry(cfg: HuntConfig) -> Gantry:
	"""Factory: fake when cfg.fake else real serial Zaber."""
	if cfg.fake:
		g: Gantry = FakeGantry(cfg.workspace, cfg.settle_tol_mm)
	else:
		g = ZaberGantry(cfg)
	g.connect()
	return g
