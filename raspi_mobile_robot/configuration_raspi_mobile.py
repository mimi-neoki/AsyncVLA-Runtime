from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RaspiMobileRobotConfig:
    camera_index: int = 0
    image_key: str = "front_image"
    timestamp_key: str = "timestamp_ns"
    goal_pose_key: str = "goal_pose"
    current_pose_key: str = "current_pose"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 15.0
