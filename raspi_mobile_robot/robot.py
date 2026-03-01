from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .configuration_raspi_mobile import RaspiMobileRobotConfig

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from lerobot.common.robots.robot import Robot
except Exception:  # pragma: no cover
    class Robot:
        def connect(self) -> None:  # noqa: D401
            """Connect to hardware."""
            return

        def disconnect(self) -> None:  # noqa: D401
            """Disconnect from hardware."""
            return


@dataclass
class TwistCommand:
    linear: float
    angular: float


class RaspiMobileRobot(Robot):
    """LeRobot-compatible mobile robot wrapper for Raspberry Pi edge runtime."""

    def __init__(
        self,
        config: RaspiMobileRobotConfig,
        image_provider: Callable[[], np.ndarray] | None = None,
        odom_provider: Callable[[], np.ndarray] | None = None,
        goal_pose_provider: Callable[[], np.ndarray] | None = None,
        cmd_vel_publisher: Callable[[TwistCommand], None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._image_provider = image_provider
        self._odom_provider = odom_provider
        self._goal_pose_provider = goal_pose_provider
        self._cmd_vel_publisher = cmd_vel_publisher
        self._cap: Any = None
        self._last_command = TwistCommand(0.0, 0.0)

    def connect(self) -> None:
        if self._image_provider is not None:
            return
        if cv2 is None:
            raise RuntimeError("OpenCV not available; provide image_provider explicitly")
        self._cap = cv2.VideoCapture(self.config.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_image(self) -> np.ndarray:
        if self._image_provider is not None:
            return np.asarray(self._image_provider())
        if self._cap is None:
            raise RuntimeError("Robot camera is not connected")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame")
        return frame

    def get_observation(self) -> dict[str, Any]:
        frame = self._capture_image()
        ts = time.monotonic_ns()
        obs: dict[str, Any] = {
            self.config.image_key: frame,
            "images": {self.config.image_key: frame},
            self.config.timestamp_key: ts,
        }
        if self._odom_provider is not None:
            obs[self.config.current_pose_key] = np.asarray(self._odom_provider(), dtype=np.float32)
        if self._goal_pose_provider is not None:
            obs[self.config.goal_pose_key] = np.asarray(self._goal_pose_provider(), dtype=np.float32)
        return obs

    def send_action(self, action: dict[str, Any] | np.ndarray | list[float]) -> None:
        if isinstance(action, dict):
            linear = float(action.get("linear", action.get("linear_x", 0.0)))
            angular = float(action.get("angular", action.get("angular_z", 0.0)))
        else:
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.size < 2:
                raise ValueError("Action requires at least [linear, angular]")
            linear, angular = float(values[0]), float(values[1])
        cmd = TwistCommand(linear=linear, angular=angular)
        self._last_command = cmd
        if self._cmd_vel_publisher is not None:
            self._cmd_vel_publisher(cmd)

    @property
    def last_command(self) -> TwistCommand:
        return self._last_command
