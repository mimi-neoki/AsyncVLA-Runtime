from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.benchmark_base_vla_server import _build_payload_template


def _make_args(*, task_mode: str | None, instruction: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        input_profile="official_demo",
        goal_x=1.0,
        goal_y=-2.0,
        goal_yaw=90.0,
        metric_waypoint_spacing=0.1,
        instruction=instruction,
        task_mode=task_mode,
        task_id=None,
        satellite=None,
        image_key="front_image",
        goal_image_key="goal_image",
    )


def test_build_payload_template_language_only_omits_goal_pose() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image_blob = {"encoding": "jpeg_base64", "data": "abc", "shape": list(image.shape)}

    payload = _build_payload_template(
        _make_args(task_mode="language_only", instruction="move to the bottle"),
        image_rgb=image,
        image_blob=image_blob,
        goal_blob=None,
    )

    assert payload["instruction"] == "move to the bottle"
    assert payload["task_mode"] == "language_only"
    assert "goal_pose" not in payload


def test_build_payload_template_pose_only_keeps_goal_pose() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image_blob = {"encoding": "jpeg_base64", "data": "abc", "shape": list(image.shape)}

    payload = _build_payload_template(
        _make_args(task_mode="pose_only"),
        image_rgb=image,
        image_blob=image_blob,
        goal_blob=None,
    )

    assert payload["task_mode"] == "pose_only"
    assert payload["goal_pose"][0] == 10.0
    assert payload["goal_pose"][1] == -20.0
    assert abs(payload["goal_pose"][2]) < 1e-6
    assert abs(payload["goal_pose"][3] - 1.0) < 1e-6
