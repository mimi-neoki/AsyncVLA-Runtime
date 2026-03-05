from __future__ import annotations

import numpy as np

from asyncvla_pi.edge_robot_client import EdgeAwareRobotClient, EdgeRobotClientConfig


def _build_client(config: EdgeRobotClientConfig) -> EdgeAwareRobotClient:
    client = EdgeAwareRobotClient(
        robot=object(),
        edge_runner=object(),
        pd_controller=object(),
        config=config,
    )
    client._encode_image = lambda image: "encoded-image"  # type: ignore[method-assign]
    return client


def test_payload_uses_defaults_for_instruction_and_task_fields() -> None:
    config = EdgeRobotClientConfig(
        policy_url="http://127.0.0.1:8000/infer",
        default_instruction="go to the target",
        default_task_mode="language_and_pose",
        default_task_id=8,
        default_satellite=False,
    )
    client = _build_client(config)

    observation = {
        "front_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "goal_pose": [1.0, 2.0, 0.3],
        "current_pose": [0.0, 0.0, 0.0],
        "timestamp_ns": 100,
    }
    payload = client._build_policy_payload(observation)

    assert payload["instruction"] == "go to the target"
    assert payload["task_mode"] == "language_and_pose"
    assert payload["task_id"] == 8
    assert payload["satellite"] is False


def test_payload_prioritizes_observation_fields_over_defaults() -> None:
    config = EdgeRobotClientConfig(
        policy_url="http://127.0.0.1:8000/infer",
        default_instruction="default instruction",
        default_task_mode="pose_only",
        default_task_id=4,
        default_satellite=False,
    )
    client = _build_client(config)

    observation = {
        "front_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "goal_pose": [1.0, 2.0, 0.3],
        "current_pose": [0.0, 0.0, 0.0],
        "instruction": "edge override",
        "task_mode": "all",
        "task_id": 3,
        "satellite": True,
    }
    payload = client._build_policy_payload(observation)

    assert payload["instruction"] == "edge override"
    assert payload["task_mode"] == "all"
    assert payload["task_id"] == 3
    assert payload["satellite"] is True
