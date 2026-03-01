from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PDControllerConfig:
    kp_linear: float = 1.0
    kd_linear: float = 0.1
    kp_heading: float = 2.0
    kd_heading: float = 0.15
    kp_yaw: float = 1.0
    max_linear: float = 0.6
    max_angular: float = 1.8
    dt_min: float = 1e-3


class PDController:
    """Converts pose targets from edge adapter to cmd_vel-style outputs."""

    def __init__(self, config: PDControllerConfig | None = None) -> None:
        self.config = config or PDControllerConfig()
        self._prev_ts_ns: int | None = None
        self._prev_distance = 0.0
        self._prev_heading_err = 0.0

    def reset(self) -> None:
        self._prev_ts_ns = None
        self._prev_distance = 0.0
        self._prev_heading_err = 0.0

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        while rad > math.pi:
            rad -= 2.0 * math.pi
        while rad < -math.pi:
            rad += 2.0 * math.pi
        return rad

    def _dt(self, timestamp_ns: int | None) -> float:
        now_ns = timestamp_ns or time.monotonic_ns()
        if self._prev_ts_ns is None:
            self._prev_ts_ns = now_ns
            return self.config.dt_min
        dt = max((now_ns - self._prev_ts_ns) / 1e9, self.config.dt_min)
        self._prev_ts_ns = now_ns
        return dt

    def compute_cmd(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        timestamp_ns: int | None = None,
    ) -> dict[str, float]:
        current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
        target = np.asarray(target_pose, dtype=np.float32).reshape(-1)
        if current.size < 3 or target.size < 3:
            raise ValueError("current_pose and target_pose must include [x, y, yaw]")

        dx = float(target[0] - current[0])
        dy = float(target[1] - current[1])
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_err = self._wrap_angle(desired_heading - float(current[2]))
        yaw_err = self._wrap_angle(float(target[2] - current[2]))

        dt = self._dt(timestamp_ns)
        d_distance = (distance - self._prev_distance) / dt
        d_heading = (heading_err - self._prev_heading_err) / dt

        linear = self.config.kp_linear * distance + self.config.kd_linear * d_distance
        angular = (
            self.config.kp_heading * heading_err
            + self.config.kd_heading * d_heading
            + self.config.kp_yaw * yaw_err
        )

        self._prev_distance = distance
        self._prev_heading_err = heading_err

        linear = float(np.clip(linear, -self.config.max_linear, self.config.max_linear))
        angular = float(np.clip(angular, -self.config.max_angular, self.config.max_angular))
        return {"linear": linear, "angular": angular}

    def cmd_from_pose_chunk(
        self,
        current_pose: np.ndarray,
        pose_chunk: np.ndarray,
        timestamp_ns: int | None = None,
    ) -> dict[str, float]:
        chunk = np.asarray(pose_chunk, dtype=np.float32)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[-1] < 3:
            raise ValueError("pose_chunk must have shape [T, >=3] or [B, T, >=3]")
        target = chunk[0, :3]
        return self.compute_cmd(current_pose=current_pose, target_pose=target, timestamp_ns=timestamp_ns)
