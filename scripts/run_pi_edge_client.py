#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import (
    EdgeAwareRobotClient,
    EdgeRobotClientConfig,
    HailoEdgeRunner,
    HailoEdgeRunnerConfig,
    PDController,
    PDControllerConfig,
)
from raspi_mobile_robot import RaspiMobileRobot, RaspiMobileRobotConfig, TwistCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pi-side edge adapter client and connect to base-VLA server")
    parser.add_argument("--policy-url", required=True, help="e.g. http://<gpu-server-ip>:8000/infer")
    parser.add_argument("--hef", default="models/edge_adapter_v520.hef")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--edge-hz", type=float, default=8.0)
    parser.add_argument("--policy-hz", type=float, default=8.0)
    parser.add_argument("--cmd-hz", type=float, default=10.0)
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)

    # HEF I/O names for current model in models/edge_adapter_v520.hef
    parser.add_argument("--input-current-name", default="edge/input_layer2")
    parser.add_argument("--input-delayed-name", default="edge/input_layer1")
    parser.add_argument("--input-tokens-name", default="edge/input_layer3")
    parser.add_argument("--output-chunk-name", default="edge/depth_to_space1")

    parser.add_argument("--image-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--input-format", choices=["uint8", "float32", "auto"], default="uint8")
    parser.add_argument("--output-format", choices=["uint8", "float32", "auto"], default="float32")
    parser.add_argument("--normalize-imagenet", action="store_true")
    parser.add_argument("--image-scale-255", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hef_path = Path(args.hef).expanduser().resolve()
    if not hef_path.exists():
        raise FileNotFoundError(f"HEF not found: {hef_path}")

    goal_pose = np.array([args.goal_x, args.goal_y, args.goal_yaw], dtype=np.float32)

    def odom_provider() -> np.ndarray:
        # Replace with real odometry integration (e.g., ROS odom).
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def goal_provider() -> np.ndarray:
        return goal_pose

    def cmd_publisher(cmd: TwistCommand) -> None:
        # Replace with real actuator publish (e.g., ROS /cmd_vel).
        print(f"cmd_vel linear={cmd.linear:.3f}, angular={cmd.angular:.3f}")

    robot = RaspiMobileRobot(
        config=RaspiMobileRobotConfig(
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        ),
        odom_provider=odom_provider,
        goal_pose_provider=goal_provider,
        cmd_vel_publisher=cmd_publisher,
    )

    edge_runner = HailoEdgeRunner(
        HailoEdgeRunnerConfig(
            hef_path=str(hef_path),
            input_current_image=args.input_current_name,
            input_delayed_image=args.input_delayed_name,
            input_projected_tokens=args.input_tokens_name,
            input_goal_pose=None,
            output_action_chunk=args.output_chunk_name,
            image_height=96,
            image_width=96,
            chunk_size=8,
            pose_dim=4,
            image_layout=args.image_layout,
            input_format_type=args.input_format,
            output_format_type=args.output_format,
            normalize_imagenet=args.normalize_imagenet,
            image_scale_255=args.image_scale_255,
        )
    )

    client = EdgeAwareRobotClient(
        robot=robot,
        edge_runner=edge_runner,
        pd_controller=PDController(PDControllerConfig()),
        config=EdgeRobotClientConfig(
            policy_url=args.policy_url,
            camera_hz=args.camera_fps,
            edge_hz=args.edge_hz,
            policy_hz=args.policy_hz,
        ),
    )

    print("Pi edge client start")
    print(f"policy_url={args.policy_url}")
    print(f"hef={hef_path}")
    print(
        f"inputs(current, delayed, tokens)=({args.input_current_name}, {args.input_delayed_name}, {args.input_tokens_name}), "
        f"output={args.output_chunk_name}"
    )
    client.run_forever()


if __name__ == "__main__":
    main()
