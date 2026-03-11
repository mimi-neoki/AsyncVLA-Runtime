#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import HailoEdgeRunner, HailoEdgeRunnerConfig, ImageRingBuffer, PDController, PDControllerConfig
from raspi_mobile_robot import RaspiMobileRobot, RaspiMobileRobotConfig

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import busy_wait

try:
    from test_yolo_world_camera import (
        YoloWorldHailoRunner,
        YoloWorldHailoRunnerConfig,
        _build_text_embeddings_with_clip,
    )
except Exception:
    from scripts.test_yolo_world_camera import (
        YoloWorldHailoRunner,
        YoloWorldHailoRunnerConfig,
        _build_text_embeddings_with_clip,
    )

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PANEL_WIDTH = 420
FPS = 50


@dataclass
class ObjectCheckResult:
    target_text: str
    present: bool
    best_score: float
    threshold: float
    infer_ms: float
    checked_at_monotonic: float
    error: str | None = None


@dataclass
class StdinObjectCheckerConfig:
    yolo_hef: str
    clip_hef: str
    clip_model_id: str
    prompt_template: str
    conf_thres: float
    iou_thres: float
    max_det: int
    timeout_ms: int
    input_image_name: str | None
    input_text_name: str | None


class StdinObjectChecker:
    def __init__(self, config: StdinObjectCheckerConfig) -> None:
        self.config = config
        self._target_ctx: Any | None = None
        self._target: Any | None = None
        self._owns_target = False
        self._runner: YoloWorldHailoRunner | None = None
        self._text_embedding_cache: dict[str, np.ndarray] = {}

    def start(self, target: Any | None = None) -> None:
        from hailo_platform import VDevice

        if target is None:
            self._target_ctx = VDevice()
            self._target = self._target_ctx.__enter__()
            self._owns_target = True
        else:
            self._target = target
            self._owns_target = False
        self._runner = YoloWorldHailoRunner(
            target=self._target,
            config=YoloWorldHailoRunnerConfig(
                hef_path=self.config.yolo_hef,
                input_image_name=self.config.input_image_name,
                input_text_name=self.config.input_text_name,
                timeout_ms=self.config.timeout_ms,
            ),
        )
        self._runner.__enter__()

    def close(self) -> None:
        if self._runner is not None:
            self._runner.__exit__(None, None, None)
            self._runner = None
        if self._owns_target and self._target_ctx is not None:
            self._target_ctx.__exit__(None, None, None)
            self._target_ctx = None
        self._target = None
        self._owns_target = False

    def _text_embedding_for(self, target_text: str) -> np.ndarray:
        cache_key = target_text.strip().lower()
        if not cache_key:
            raise ValueError("Empty target text is not allowed")
        if cache_key not in self._text_embedding_cache:
            if self._target is None or self._runner is None:
                raise RuntimeError("StdinObjectChecker is not started")
            emb = _build_text_embeddings_with_clip(
                target=self._target,
                clip_hef_path=Path(self.config.clip_hef),
                yolo_infer_model=self._runner.infer_model,
                yolo_text_input_name=self._runner.text_input_name,
                class_texts=[target_text],
                clip_model_id=self.config.clip_model_id,
                prompt_template=self.config.prompt_template,
                timeout_ms=self.config.timeout_ms,
            )
            self._text_embedding_cache[cache_key] = emb
        return self._text_embedding_cache[cache_key]

    def check_once(self, frame_bgr: np.ndarray, target_text: str) -> ObjectCheckResult:
        if self._runner is None:
            raise RuntimeError("StdinObjectChecker is not started")

        text = target_text.strip()
        if not text:
            raise ValueError("Empty target text is not allowed")

        emb = self._text_embedding_for(text)
        self._runner.set_text_embeddings(emb)
        _, scores, _, infer_ms = self._runner.infer(
            frame_bgr=frame_bgr,
            num_classes=1,
            conf_thres=0.0,
            iou_thres=self.config.iou_thres,
            max_det=self.config.max_det,
        )
        best_score = float(np.max(scores)) if scores.size > 0 else 0.0
        present = best_score >= float(self.config.conf_thres)
        return ObjectCheckResult(
            target_text=text,
            present=present,
            best_score=best_score,
            threshold=float(self.config.conf_thres),
            infer_ms=float(infer_ms),
            checked_at_monotonic=time.monotonic(),
            error=None,
        )


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
    parser.add_argument(
        "--instruction-verb",
        default="move to",
        help="Verb phrase used for instruction composition (fixed during runtime).",
    )
    parser.add_argument(
        "--instruction-noun",
        default="the target object",
        help="Initial target noun phrase. You can update this via stdin while running.",
    )
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
        "--stdin-object-check",
        action="store_true",
        help="Run one-shot YOLO-World check when stdin noun phrase is updated.",
    )
    parser.add_argument("--yolo-hef", default="models/yolo_world_v2s.hef")
    parser.add_argument("--clip-hef", default="models/clip_vit_b_32_text_encoder.hef")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--yolo-prompt-template", default="a photo of {}")
    parser.add_argument("--yolo-conf-thres", type=float, default=0.50)
    parser.add_argument("--yolo-iou-thres", type=float, default=0.45)
    parser.add_argument("--yolo-max-det", type=int, default=100)
    parser.add_argument("--yolo-timeout-ms", type=int, default=10000)
    parser.add_argument("--yolo-input-image-name", default=None)
    parser.add_argument("--yolo-input-text-name", default=None)

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


def _compose_instruction(verb: str, noun: str) -> str:
    verb_text = str(verb).strip()
    noun_text = str(noun).strip()
    if verb_text and noun_text:
        return f"{verb_text} {noun_text}"
    if noun_text:
        return noun_text
    return verb_text


def _draw_overlay(
    frame_bgr: np.ndarray,
    pose_chunk: np.ndarray,
    cmd: dict[str, float],
    instruction: str,
    loop_ms: float,
    policy_ms: float,
    edge_ms: float,
    show_object_check: bool = False,
    object_check_result: ObjectCheckResult | None = None,
    object_check_active_target: str | None = None,
    object_check_pending_target: str | None = None,
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

    draw_text_fit("inference_instruction:", panel_x + 12, y0, (0, 255, 255))
    y0 += 26
    for line in textwrap.wrap(instruction, width=32)[:3]:
        draw_text_fit(line, panel_x + 12, y0, (180, 255, 180))
        y0 += 24

    if show_object_check:
        draw_text_fit("stdin_object_check:", panel_x + 12, y0, (0, 255, 255))
        y0 += 24
        if object_check_active_target:
            draw_text_fit(f"running: {object_check_active_target}", panel_x + 12, y0, (0, 200, 255))
            y0 += 22
        elif object_check_pending_target:
            draw_text_fit(f"queued: {object_check_pending_target}", panel_x + 12, y0, (0, 200, 255))
            y0 += 22

        if object_check_result is not None:
            target_line = f"last target: {object_check_result.target_text}"
            for line in textwrap.wrap(target_line, width=32)[:2]:
                draw_text_fit(line, panel_x + 12, y0, (180, 255, 180))
                y0 += 22
            if object_check_result.error:
                draw_text_fit("last result: ERROR", panel_x + 12, y0, (0, 120, 255))
                y0 += 22
            else:
                status_text = "FOUND" if object_check_result.present else "NOT FOUND"
                status_color = (80, 255, 120) if object_check_result.present else (80, 120, 255)
                draw_text_fit(f"last result: {status_text}", panel_x + 12, y0, status_color)
                y0 += 22
                draw_text_fit(
                    f"score={object_check_result.best_score:.2f} th={object_check_result.threshold:.2f}",
                    panel_x + 12,
                    y0,
                    (220, 220, 220),
                )
                y0 += 22

    # Top-down mini map in side panel
    map_y = max(110, y0 + 8)
    map_size = min(260, h - map_y - 12)
    map_size = max(120, map_size)
    pad = 12
    x1 = panel_x + (panel_w - map_size) // 2
    y1 = map_y
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

    lekiwi_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="my_lekiwi", has_arm=False)
    lekiwi = LeKiwiClient(lekiwi_config)

    shared_hailo_target_ctx: Any | None = None
    shared_hailo_target: Any | None = None
    if args.stdin_object_check:
        from hailo_platform import VDevice

        shared_hailo_target_ctx = VDevice()
        shared_hailo_target = shared_hailo_target_ctx.__enter__()

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
        ),
        target=shared_hailo_target,
    )

    ring = ImageRingBuffer(capacity=args.ring_capacity)
    pd = PDController(PDControllerConfig())
    max_delta_ns = int(args.nearest_frame_max_delta_ms * 1e6)

    lock = threading.Lock()
    hailo_lock = threading.Lock()
    running = threading.Event()
    running.set()
    latest_obs: dict[str, Any] | None = None
    latest_tokens: np.ndarray | None = None
    latest_tokens_ts: int | None = None
    last_policy_ms = 0.0
    last_policy_error: str | None = None
    current_instruction_noun = args.instruction_noun
    pending_object_check_noun: str | None = None
    active_object_check_noun: str | None = None
    last_object_check_result: ObjectCheckResult | None = None

    object_checker: StdinObjectChecker | None = None
    if args.stdin_object_check:
        yolo_hef = Path(args.yolo_hef).expanduser().resolve()
        clip_hef = Path(args.clip_hef).expanduser().resolve()
        if not yolo_hef.exists():
            raise FileNotFoundError(f"YOLO HEF not found: {yolo_hef}")
        if not clip_hef.exists():
            raise FileNotFoundError(f"CLIP HEF not found: {clip_hef}")
        object_checker = StdinObjectChecker(
            StdinObjectCheckerConfig(
                yolo_hef=str(yolo_hef),
                clip_hef=str(clip_hef),
                clip_model_id=args.clip_model_id,
                prompt_template=args.yolo_prompt_template,
                conf_thres=args.yolo_conf_thres,
                iou_thres=args.yolo_iou_thres,
                max_det=args.yolo_max_det,
                timeout_ms=args.yolo_timeout_ms,
                input_image_name=args.yolo_input_image_name,
                input_text_name=args.yolo_input_text_name,
            )
        )
        object_checker.start(target=shared_hailo_target)

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
    print(f"instruction_verb={args.instruction_verb}")
    print(f"instruction_noun={args.instruction_noun}")
    print(f"instruction={_compose_instruction(args.instruction_verb, args.instruction_noun)}")
    print("Type a new noun phrase and press Enter to update it while running.")
    if args.stdin_object_check:
        print(
            "stdin_object_check=enabled "
            f"(conf_thres={args.yolo_conf_thres:.2f}, iou_thres={args.yolo_iou_thres:.2f})"
        )
    if args.task_mode is not None:
        print(f"task_mode={args.task_mode}")
    if args.task_id is not None:
        print(f"task_id={args.task_id}")
    if args.satellite is not None:
        print(f"satellite={args.satellite}")
    robot.connect()
    lekiwi.connect()
    assert lekiwi.is_connected, "Failed to connect to LeKiwiClient"

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
            instruction_noun: str
            with lock:
                obs = dict(latest_obs) if latest_obs is not None else None
                instruction_noun = str(current_instruction_noun)
            instruction_text = _compose_instruction(args.instruction_verb, instruction_noun)
            if obs is not None:
                try:
                    frame = np.asarray(obs["front_image"])
                    ts = int(obs["timestamp_ns"])
                    payload = {
                        "timestamp_ns": ts,
                        "instruction": instruction_text,
                        "goal_pose": np.asarray(obs.get("goal_pose", [0, 0, 0]), dtype=np.float32).tolist(),
                        "current_pose": np.asarray(obs.get("current_pose", [0, 0, 0]), dtype=np.float32).tolist(),
                        "images": {
                            "front_image": {
                                "encoding": "jpeg_base64",
                                "data": _encode_jpeg_base64(frame, args.jpeg_quality),
                                "shape": list(frame.shape),
                            }
                        },
                    }
                    if args.task_mode is not None:
                        payload["task_mode"] = args.task_mode
                    if args.task_id is not None:
                        payload["task_id"] = int(args.task_id)
                    if args.satellite is not None:
                        payload["satellite"] = bool(args.satellite)
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

    def instruction_input_loop() -> None:
        nonlocal current_instruction_noun, pending_object_check_noun
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            text = line.strip()
            if not text:
                continue
            with lock:
                current_instruction_noun = text
                if args.stdin_object_check:
                    pending_object_check_noun = text
            if args.stdin_object_check:
                print(f"[instruction noun updated] {text} (object check queued)")
            else:
                print(f"[instruction noun updated] {text}")

    def object_check_loop() -> None:
        nonlocal pending_object_check_noun, active_object_check_noun, last_object_check_result
        if object_checker is None:
            return
        while running.is_set():
            target_text: str | None = None
            frame_bgr: np.ndarray | None = None
            with lock:
                if pending_object_check_noun is not None and latest_obs is not None and active_object_check_noun is None:
                    target_text = str(pending_object_check_noun)
                    pending_object_check_noun = None
                    active_object_check_noun = target_text
                    frame_bgr = np.asarray(latest_obs["front_image"]).copy()
            if target_text is None or frame_bgr is None:
                time.sleep(0.02)
                continue

            try:
                with hailo_lock:
                    result = object_checker.check_once(frame_bgr=frame_bgr, target_text=target_text)
                print(
                    "[stdin object check] "
                    f"target='{result.target_text}' present={result.present} "
                    f"best_score={result.best_score:.3f} th={result.threshold:.3f} infer_ms={result.infer_ms:.1f}"
                )
            except Exception as exc:
                result = ObjectCheckResult(
                    target_text=target_text,
                    present=False,
                    best_score=0.0,
                    threshold=args.yolo_conf_thres,
                    infer_ms=0.0,
                    checked_at_monotonic=time.monotonic(),
                    error=str(exc),
                )
                print(f"[stdin object check error] target='{target_text}' error={exc}")

            with lock:
                last_object_check_result = result
                active_object_check_noun = None

    capture_thread = threading.Thread(target=capture_loop, name="capture", daemon=True)
    policy_thread = threading.Thread(target=policy_loop, name="policy", daemon=True)
    instruction_thread = threading.Thread(target=instruction_input_loop, name="instruction_input", daemon=True)
    object_check_thread: threading.Thread | None = None
    if object_checker is not None:
        object_check_thread = threading.Thread(target=object_check_loop, name="object_check", daemon=True)
    capture_thread.start()
    policy_thread.start()
    instruction_thread.start()
    if object_check_thread is not None:
        object_check_thread.start()

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
                instruction_noun = str(current_instruction_noun)
                object_check_result = last_object_check_result
                object_check_active = active_object_check_noun
                object_check_pending = pending_object_check_noun
            instruction_text = _compose_instruction(args.instruction_verb, instruction_noun)

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
                    with hailo_lock:
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
            lekiwi_action = lekiwi._from_bi_wheel_action_to_base_action(cmd)
            # print(lekiwi_action)
            lekiwi.send_action(lekiwi_action)

            loop_ms = (time.monotonic() - t0) * 1000.0
            vis = _draw_overlay(
                frame_bgr=current,
                pose_chunk=pose_chunk,
                cmd=cmd,
                instruction=instruction_text,
                loop_ms=loop_ms,
                policy_ms=policy_ms,
                edge_ms=edge_ms,
                show_object_check=args.stdin_object_check,
                object_check_result=object_check_result,
                object_check_active_target=object_check_active,
                object_check_pending_target=object_check_pending,
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
        instruction_thread.join(timeout=1.0)
        if object_check_thread is not None:
            object_check_thread.join(timeout=5.0)
        edge_runner.close()
        robot.disconnect()
        lekiwi.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        lekiwi.disconnect()
        if object_checker is not None:
            object_checker.close()
        if shared_hailo_target_ctx is not None:
            shared_hailo_target_ctx.__exit__(None, None, None)
        if writer is not None:
            writer.release()
        if args.show and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
