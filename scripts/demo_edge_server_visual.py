#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import HailoEdgeRunner, HailoEdgeRunnerConfig, ImageRingBuffer, PDController, PDControllerConfig
from asyncvla_pi.policy_payload import build_policy_payload
from raspi_mobile_robot import RaspiMobileRobot, RaspiMobileRobotConfig

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PANEL_WIDTH = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: camera -> base VLA server -> Hailo edge adapter -> action overlay"
    )
    parser.add_argument("--policy-url", required=True, help="e.g. http://<server-ip>:8000/infer")
    parser.add_argument("--hef", default="models/edge_adapter_v520.hef")
    parser.add_argument("--camera-index", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=60.0)
    parser.add_argument("--loop-hz", type=float, default=8.0, help="Edge/draw loop frequency")
    parser.add_argument("--policy-hz", type=float, default=8.0, help="Server communication loop frequency")
    parser.add_argument("--policy-timeout", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--ring-capacity", type=int, default=256)
    parser.add_argument("--nearest-frame-max-delta-ms", type=float, default=250.0)
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)
    parser.add_argument("--instruction", default="move forward")
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

    # HEF vstream names for edge_adapter_v520.hef
    parser.add_argument("--input-current-name", default="edge/input_layer2")
    parser.add_argument("--input-delayed-name", default="edge/input_layer1")
    parser.add_argument("--input-tokens-name", default="edge/input_layer3")
    parser.add_argument("--output-chunk-name", default="edge/depth_to_space1")
    parser.add_argument("--image-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--input-format", choices=["uint8", "float32", "auto"], default="uint8")
    parser.add_argument("--output-format", choices=["uint8", "float32", "auto"], default="auto")
    parser.add_argument("--normalize-imagenet", action="store_true")
    parser.add_argument("--image-scale-255", action="store_true")

    parser.add_argument("--show", action="store_true", help="Show OpenCV window")
    parser.add_argument("--save-video", default="", help="Optional output video path")

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


def _parse_camera_index(raw: str) -> int | str:
    if raw.startswith("/dev/video"):
        return raw
    return int(raw)


def _encode_jpeg_base64(image_bgr: np.ndarray, quality: int) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for JPEG encoding")
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pose_chunk_to_matrix(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[0] == 8:
        arr = arr.reshape(8, -1)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(8, -1)
    return arr.astype(np.float32)


def _target_pose_from_action(action_step: np.ndarray) -> np.ndarray:
    step = np.asarray(action_step, dtype=np.float32).reshape(-1)
    x = float(step[0]) if step.size > 0 else 0.0
    y = float(step[1]) if step.size > 1 else 0.0
    if step.size >= 4:
        yaw = float(np.arctan2(step[3], step[2]))
    elif step.size >= 3:
        yaw = float(step[2])
    else:
        yaw = 0.0
    return np.array([x, y, yaw], dtype=np.float32)


def _draw_overlay(
    frame_bgr: np.ndarray,
    pose_chunk: np.ndarray,
    cmd: dict[str, float],
    loop_ms: float,
    policy_ms: float,
    edge_ms: float,
    error_text: str | None = None,
) -> np.ndarray:
    if cv2 is None:
        return frame_bgr

    camera = frame_bgr.copy()
    h, w = camera.shape[:2]
    panel_w = PANEL_WIDTH
    vis = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    vis[:, :w] = camera
    panel_x = w

    # HUD text
    texts = [
        f"loop={loop_ms:.1f}ms policy={policy_ms:.1f}ms edge={edge_ms:.1f}ms",
        f"linear={cmd.get('linear', 0.0):+.3f} angular={cmd.get('angular', 0.0):+.3f}",
    ]
    if error_text:
        texts.append(f"error={error_text}")

    cv2.rectangle(vis, (panel_x, 0), (w + panel_w, h), (26, 26, 26), thickness=-1)
    cv2.rectangle(vis, (panel_x, 0), (w + panel_w - 1, h - 1), (80, 80, 80), thickness=1)

    def draw_text_fit(text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        max_w = panel_w - 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        scale = 0.58
        while scale > 0.34:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw <= max_w:
                break
            scale -= 0.04
        cv2.putText(vis, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    y0 = 32
    for t in texts:
        draw_text_fit(t, panel_x + 12, y0, (0, 255, 255))
        y0 += 28

    # Top-down mini map in side panel
    map_size = min(260, h - 140)
    pad = 12
    x1 = panel_x + (panel_w - map_size) // 2
    y1 = 110
    x2 = x1 + map_size
    y2 = y1 + map_size
    cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 30, 30), thickness=-1)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), thickness=1)

    origin = (x1 + map_size // 2, y2 - 18)
    cv2.circle(vis, origin, 4, (255, 255, 255), -1)

    # Assumption: action is local trajectory [x(forward), y(left), ...].
    # Use dynamic scaling so points stay inside the panel regardless of range.
    xs = [0.0]
    ys = [0.0]
    for step in pose_chunk:
        xs.append(float(step[0]) if step.size > 0 else 0.0)
        ys.append(float(step[1]) if step.size > 1 else 0.0)

    max_abs_x = max(abs(v) for v in xs) + 1e-6
    max_abs_y = max(abs(v) for v in ys) + 1e-6
    # Keep some margin on each side.
    avail_left_right = (map_size * 0.45)
    avail_forward = (map_size * 0.78)
    sx = avail_forward / max_abs_x
    sy = avail_left_right / max_abs_y
    # Allow sufficiently small scales for large action magnitudes.
    scale = max(0.05, min(sx, sy))

    pts: list[tuple[int, int]] = [origin]
    for x, y in zip(xs[1:], ys[1:]):
        px = int(origin[0] - y * scale)
        py = int(origin[1] - x * scale)
        # Clip to map area to avoid drawing outside panel
        px = max(x1 + 2, min(x2 - 2, px))
        py = max(y1 + 2, min(y2 - 2, py))
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(vis, pts[i - 1], pts[i], (80, 220, 80), 2)
    for p in pts[1:]:
        cv2.circle(vis, p, 2, (30, 180, 255), -1)

    cv2.putText(vis, "Action chunk (top view)", (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(
        vis,
        f"auto-scale={scale:.2f}",
        (x1 + 8, y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(vis, "Camera", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required")
    _ensure_hailo_runtime_available()

    libcamerify_active = os.environ.get("ASYNCVLA_LIBCAMERIFY_ACTIVE") == "1"
    if args.libcamerify == "on" and not libcamerify_active:
        _reexec_with_libcamerify()
    if args.libcamerify == "auto" and not libcamerify_active and shutil.which("libcamerify"):
        _reexec_with_libcamerify()

    hef_path = Path(args.hef).expanduser().resolve()
    if not hef_path.exists():
        raise FileNotFoundError(f"HEF not found: {hef_path}")

    robot = RaspiMobileRobot(
        config=RaspiMobileRobotConfig(
            camera_index=_parse_camera_index(args.camera_index),
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        ),
        odom_provider=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        goal_pose_provider=lambda: np.array([args.goal_x, args.goal_y, args.goal_yaw], dtype=np.float32),
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

    ring = ImageRingBuffer(capacity=args.ring_capacity)
    pd = PDController(PDControllerConfig())
    max_delta_ns = int(args.nearest_frame_max_delta_ms * 1e6)

    lock = threading.Lock()
    running = threading.Event()
    running.set()
    latest_obs: dict[str, Any] | None = None
    latest_tokens: np.ndarray | None = None
    latest_tokens_ts: int | None = None
    last_policy_ms = 0.0
    last_policy_error: str | None = None

    writer: cv2.VideoWriter | None = None
    if args.save_video:
        out_path = Path(args.save_video).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            max(args.loop_hz, 1.0),
            (args.camera_width + PANEL_WIDTH, args.camera_height),
        )

    print(f"Start demo. policy_url={args.policy_url}, hef={hef_path}")
    print(f"instruction={args.instruction}")
    print(f"metric_waypoint_spacing={args.metric_waypoint_spacing}")
    if args.task_mode is not None:
        print(f"task_mode={args.task_mode}")
    if args.task_id is not None:
        print(f"task_id={args.task_id}")
    if args.satellite is not None:
        print(f"satellite={args.satellite}")
    robot.connect()

    def capture_loop() -> None:
        nonlocal latest_obs
        period = 1.0 / max(args.camera_fps, 1e-6)
        while running.is_set():
            t0 = time.monotonic()
            try:
                obs = robot.get_observation()
                frame = np.asarray(obs["front_image"])
                ts = int(obs["timestamp_ns"])
                ring.push(frame, ts, metadata={"observation": obs})
                with lock:
                    latest_obs = obs
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def policy_loop() -> None:
        nonlocal latest_tokens, latest_tokens_ts, last_policy_ms, last_policy_error
        period = 1.0 / max(args.policy_hz, 1e-6)
        while running.is_set():
            t0 = time.monotonic()
            obs: dict[str, Any] | None
            with lock:
                obs = dict(latest_obs) if latest_obs is not None else None
            if obs is not None:
                try:
                    frame = np.asarray(obs["front_image"])
                    ts = int(obs["timestamp_ns"])
                    payload = build_policy_payload(
                        image=frame,
                        encoded_image=_encode_jpeg_base64(frame, args.jpeg_quality),
                        timestamp_ns=ts,
                        current_pose=obs.get("current_pose", [0, 0, 0]),
                        goal_pose=obs.get("goal_pose"),
                        instruction=args.instruction,
                        task_mode=args.task_mode,
                        task_id=args.task_id,
                        satellite=args.satellite,
                        metric_waypoint_spacing=args.metric_waypoint_spacing,
                    )
                    t_policy = time.monotonic()
                    resp = requests.post(
                        args.policy_url,
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                        timeout=args.policy_timeout,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    tokens = np.asarray(body["projected_tokens"], dtype=np.float32)
                    echoed_ts = int(body.get("timestamp_ns", ts))
                    with lock:
                        latest_tokens = tokens
                        latest_tokens_ts = echoed_ts
                        last_policy_ms = (time.monotonic() - t_policy) * 1000.0
                        last_policy_error = None
                except Exception as exc:
                    with lock:
                        last_policy_error = str(exc)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    capture_thread = threading.Thread(target=capture_loop, name="capture", daemon=True)
    policy_thread = threading.Thread(target=policy_loop, name="policy", daemon=True)
    capture_thread.start()
    policy_thread.start()

    edge_period = 1.0 / max(args.loop_hz, 1e-6)
    try:
        while running.is_set():
            t0 = time.monotonic()
            edge_ms = 0.0
            cmd = {"linear": 0.0, "angular": 0.0}
            pose_chunk = np.zeros((8, 4), dtype=np.float32)

            with lock:
                obs = dict(latest_obs) if latest_obs is not None else None
                tokens = None if latest_tokens is None else np.asarray(latest_tokens)
                tokens_ts = latest_tokens_ts
                policy_ms = last_policy_ms
                policy_err = last_policy_error

            error_text = policy_err
            if obs is not None:
                current = np.asarray(obs["front_image"])
            else:
                current = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)

            if obs is not None and tokens is not None and tokens_ts is not None:
                try:
                    delayed = ring.nearest(tokens_ts, max_delta_ns=max_delta_ns)
                    if delayed is None:
                        raise RuntimeError("No delayed frame in ring buffer for echoed timestamp")
                    t_edge = time.monotonic()
                    raw_out = edge_runner.infer(
                        current_image=current,
                        delayed_image=delayed.frame,
                        projected_tokens=tokens,
                        goal_pose=np.asarray(obs.get("goal_pose", [0, 0, 0]), dtype=np.float32),
                    )
                    edge_ms = (time.monotonic() - t_edge) * 1000.0
                    pose_chunk = _pose_chunk_to_matrix(raw_out)
                    target_pose = _target_pose_from_action(pose_chunk[0])
                    current_pose = np.asarray(obs.get("current_pose", [0.0, 0.0, 0.0]), dtype=np.float32)
                    cmd = pd.compute_cmd(current_pose=current_pose, target_pose=target_pose, timestamp_ns=int(obs.get("timestamp_ns", 0)))
                except Exception as exc:
                    error_text = str(exc)

            loop_ms = (time.monotonic() - t0) * 1000.0
            vis = _draw_overlay(
                frame_bgr=current,
                pose_chunk=pose_chunk,
                cmd=cmd,
                loop_ms=loop_ms,
                policy_ms=policy_ms,
                edge_ms=edge_ms,
                error_text=error_text,
            )

            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow("asyncvla_edge_demo", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            time.sleep(max(0.0, edge_period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        running.clear()
        capture_thread.join(timeout=1.0)
        policy_thread.join(timeout=1.0)
        robot.disconnect()
        if writer is not None:
            writer.release()
        if args.show and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
