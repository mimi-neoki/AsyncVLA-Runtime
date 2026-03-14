from __future__ import annotations

import math

import numpy as np

from asyncvla_pi.policy_payload import build_goal_pose_payload, build_policy_payload


def test_build_goal_pose_payload_normalizes_xyyaw() -> None:
    payload = build_goal_pose_payload([1.0, -2.0, math.pi / 2.0], metric_waypoint_spacing=0.1)
    assert payload is not None
    assert payload[0] == 10.0
    assert payload[1] == -20.0
    assert abs(payload[2]) < 1e-6
    assert abs(payload[3] - 1.0) < 1e-6


def test_build_policy_payload_pose_only_omits_instruction() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = build_policy_payload(
        image=image,
        encoded_image="abc",
        timestamp_ns=123,
        current_pose=[0.0, 0.0, 0.0],
        goal_pose=[1.0, 2.0, 0.0],
        instruction="move to the red cup",
        task_mode="pose_only",
        metric_waypoint_spacing=0.1,
    )
    assert payload["timestamp_ns"] == 123
    assert "instruction" not in payload
    assert payload["goal_pose"] == [10.0, 20.0, 1.0, 0.0]


def test_build_policy_payload_language_only_omits_goal_pose() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = build_policy_payload(
        image=image,
        encoded_image="abc",
        goal_pose=[1.0, 2.0, 0.0],
        instruction="move to the red cup",
        task_mode="language_only",
    )
    assert payload["instruction"] == "move to the red cup"
    assert "goal_pose" not in payload


def test_build_policy_payload_auto_preserves_instruction_and_goal_pose() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = build_policy_payload(
        image=image,
        encoded_image="abc",
        goal_pose=[1.0, 2.0, 0.0],
        instruction="move to the red cup",
        task_mode=None,
    )
    assert payload["instruction"] == "move to the red cup"
    assert payload["goal_pose"] == [10.0, 20.0, 1.0, 0.0]
