#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
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
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)
    parser.add_argument("--instruction", default=None, help="Language instruction sent to base-VLA server.")
    parser.add_argument(
        "--task-mode",
        choices=[
            "auto",
            "satellite_only",
            "pose_and_satellite",
            "satellite_and_image",
            "all",
            "pose_only",
            "pose_and_image",
            "image_only",
            "language_only",
            "language_and_pose",
        ],
        default=None,
        help="Optional AsyncVLA task mode override sent to the server.",
    )
    parser.add_argument("--task-id", type=int, default=None, help="Optional AsyncVLA modality id override (0-8).")
    parser.add_argument(
        "--satellite",
        dest="satellite",
        action="store_true",
        help="Send satellite=true in policy payload.",
    )
    parser.add_argument(
        "--no-satellite",
        dest="satellite",
        action="store_false",
        help="Send satellite=false in policy payload.",
    )
    parser.set_defaults(satellite=None)

    # HEF I/O names for current model in models/edge_adapter_v520.hef
    parser.add_argument("--input-current-name", default="edge/input_layer2")
    parser.add_argument("--input-delayed-name", default="edge/input_layer1")
    parser.add_argument("--input-tokens-name", default="edge/input_layer3")
    parser.add_argument("--output-chunk-name", default="edge/depth_to_space1")

    parser.add_argument("--image-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--input-format", choices=["uint8", "float32", "auto"], default="uint8")
    parser.add_argument("--output-format", choices=["uint8", "float32", "auto"], default="auto")
    parser.add_argument("--normalize-imagenet", action="store_true")
    parser.add_argument("--image-scale-255", action="store_true")
    parser.add_argument(
        "--libcamerify",
        choices=["auto", "off", "on"],
        default="auto",
        help="Wrap process with libcamerify for OpenCV camera capture on libcamera systems.",
    )
    return parser.parse_args()


def _reexec_with_libcamerify() -> None:
    libcamerify = shutil.which("libcamerify")
    if not libcamerify:
        raise RuntimeError("libcamerify command not found")
    env = dict(os.environ)
    env["ASYNCVLA_LIBCAMERIFY_ACTIVE"] = "1"
    cmd = [libcamerify, sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    print("Re-exec with libcamerify for OpenCV camera capture...")
    os.execvpe(libcamerify, cmd, env)


def _ensure_hailo_runtime_available() -> None:
    try:
        import hailo_platform  # noqa: F401
        return
    except Exception as exc:
        already_fallback = os.environ.get("ASYNCVLA_SYSTEMPY_ACTIVE") == "1"
        system_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else shutil.which("python3")
        current_python = Path(sys.executable).resolve()
        if (
            not already_fallback
            and system_python
            and Path(system_python).resolve() != current_python
        ):
            env = dict(os.environ)
            env["ASYNCVLA_SYSTEMPY_ACTIVE"] = "1"
            cmd = [system_python, str(Path(__file__).resolve()), *sys.argv[1:]]
            print(
                "Hailo runtime import failed in current interpreter; "
                "retrying with system python3..."
            )
            os.execvpe(system_python, cmd, env)
        raise RuntimeError(
            "pyhailort is not available in this Python environment. "
            "Install matching hailort package for this interpreter or run with system python3."
        ) from exc


def main() -> None:
    args = parse_args()
    _ensure_hailo_runtime_available()
    libcamerify_active = os.environ.get("ASYNCVLA_LIBCAMERIFY_ACTIVE") == "1"
    if args.libcamerify == "on" and not libcamerify_active:
        _reexec_with_libcamerify()
    if args.libcamerify == "auto" and not libcamerify_active and shutil.which("libcamerify"):
        _reexec_with_libcamerify()

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
            default_instruction=args.instruction,
            default_task_mode=args.task_mode,
            default_task_id=args.task_id,
            default_satellite=args.satellite,
            camera_hz=args.camera_fps,
            edge_hz=args.edge_hz,
            policy_hz=args.policy_hz,
            metric_waypoint_spacing=args.metric_waypoint_spacing,
        ),
    )

    print("Pi edge client start")
    print(f"policy_url={args.policy_url}")
    print(f"hef={hef_path}")
    if args.instruction is not None:
        print(f"instruction={args.instruction}")
    if args.task_mode is not None:
        print(f"task_mode={args.task_mode}")
    if args.task_id is not None:
        print(f"task_id={args.task_id}")
    if args.satellite is not None:
        print(f"satellite={args.satellite}")
    print(f"metric_waypoint_spacing={args.metric_waypoint_spacing}")
    print(
        f"inputs(current, delayed, tokens)=({args.input_current_name}, {args.input_delayed_name}, {args.input_tokens_name}), "
        f"output={args.output_chunk_name}"
    )
    client.run_forever()


if __name__ == "__main__":
    main()
