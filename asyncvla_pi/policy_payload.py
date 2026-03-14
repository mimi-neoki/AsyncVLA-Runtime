from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

TASK_MODE_TO_ID = {
    "satellite_only": 0,
    "pose_and_satellite": 1,
    "satellite_and_image": 2,
    "all": 3,
    "pose_only": 4,
    "pose_and_image": 5,
    "image_only": 6,
    "language_only": 7,
    "language_and_pose": 8,
}

POSE_TASK_MODES = {
    "pose_and_satellite",
    "all",
    "pose_only",
    "pose_and_image",
    "language_and_pose",
}

LANGUAGE_TASK_MODES = {
    "language_only",
    "language_and_pose",
}


def canonical_task_mode(task_mode: Any = None, task_id: Any = None) -> str | None:
    if task_id is not None:
        mode_id = int(task_id)
        for name, candidate in TASK_MODE_TO_ID.items():
            if candidate == mode_id:
                return name
        raise ValueError(f"Unsupported task_id: {task_id}")

    if task_mode is None:
        return None
    mode_raw = str(task_mode).strip().lower()
    if mode_raw in {"", "none", "auto"}:
        return None
    aliases = {
        "0": "satellite_only",
        "1": "pose_and_satellite",
        "2": "satellite_and_image",
        "3": "all",
        "4": "pose_only",
        "5": "pose_and_image",
        "6": "image_only",
        "7": "language_only",
        "8": "language_and_pose",
        "satellite only": "satellite_only",
        "pose and satellite": "pose_and_satellite",
        "satellite and image": "satellite_and_image",
        "pose only": "pose_only",
        "pose and image": "pose_and_image",
        "image only": "image_only",
        "language only": "language_only",
        "language and pose": "language_and_pose",
        "satellite-only": "satellite_only",
        "pose-and-satellite": "pose_and_satellite",
        "satellite-and-image": "satellite_and_image",
        "pose-only": "pose_only",
        "pose-and-image": "pose_and_image",
        "image-only": "image_only",
        "language-only": "language_only",
        "language-and-pose": "language_and_pose",
    }
    canonical = aliases.get(mode_raw, mode_raw)
    if canonical not in TASK_MODE_TO_ID:
        raise ValueError(f"Unsupported task_mode: {task_mode!r}")
    return canonical


def build_goal_pose_payload(goal_pose: Any, metric_waypoint_spacing: float = 0.1) -> list[float] | None:
    if goal_pose is None:
        return None
    pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
    if pose.size == 0:
        return None
    spacing = float(metric_waypoint_spacing)
    if spacing <= 0.0:
        raise ValueError(f"metric_waypoint_spacing must be > 0, got {metric_waypoint_spacing}")
    if pose.size >= 4:
        return pose[:4].astype(np.float32).tolist()
    x = float(pose[0]) if pose.size >= 1 else 0.0
    y = float(pose[1]) if pose.size >= 2 else 0.0
    yaw = float(pose[2]) if pose.size >= 3 else 0.0
    return [
        x / spacing,
        y / spacing,
        math.cos(yaw),
        math.sin(yaw),
    ]


def build_policy_payload(
    *,
    image: np.ndarray,
    encoded_image: str,
    timestamp_ns: int | None = None,
    current_pose: Any = None,
    goal_pose: Any = None,
    instruction: str | None = None,
    task_mode: Any = None,
    task_id: Any = None,
    satellite: bool | None = None,
    image_key: str = "front_image",
    timestamp_key: str = "timestamp_ns",
    current_pose_key: str = "current_pose",
    goal_pose_key: str = "goal_pose",
    instruction_key: str = "instruction",
    task_mode_key: str = "task_mode",
    task_id_key: str = "task_id",
    satellite_key: str = "satellite",
    metric_waypoint_spacing: float = 0.1,
) -> dict[str, Any]:
    payload = {
        timestamp_key: int(timestamp_ns if timestamp_ns is not None else time.monotonic_ns()),
        current_pose_key: np.asarray(
            current_pose if current_pose is not None else [0.0, 0.0, 0.0], dtype=np.float32
        ).tolist(),
        "images": {
            image_key: {
                "encoding": "jpeg_base64",
                "data": encoded_image,
                "shape": list(image.shape),
            }
        },
    }

    mode_name = canonical_task_mode(task_mode=task_mode, task_id=task_id)
    clean_instruction = None if instruction is None else str(instruction).strip()
    include_instruction = bool(clean_instruction)
    include_goal_pose = goal_pose is not None

    if mode_name is not None:
        include_instruction = bool(clean_instruction) and mode_name in LANGUAGE_TASK_MODES
        include_goal_pose = goal_pose is not None and mode_name in POSE_TASK_MODES

    if include_instruction and clean_instruction:
        payload[instruction_key] = clean_instruction

    if include_goal_pose:
        goal_pose_payload = build_goal_pose_payload(goal_pose, metric_waypoint_spacing=metric_waypoint_spacing)
        if goal_pose_payload is not None:
            payload[goal_pose_key] = goal_pose_payload

    if task_mode is not None:
        payload[task_mode_key] = task_mode
    if task_id is not None:
        payload[task_id_key] = int(task_id)
    if satellite is not None:
        payload[satellite_key] = bool(satellite)
    return payload
