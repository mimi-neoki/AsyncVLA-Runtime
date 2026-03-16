#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asyncvla_pi.edge_adapter_model import load_edge_adapter_from_hf_snapshot
from asyncvla_pi.policy_payload import build_policy_payload

TASK_MODE_CHOICES = [
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
]

INPUT_PROFILE_CHOICES = [
    "official_demo",
    "runtime_generic",
]

_THREAD_LOCAL = threading.local()


@dataclass
class RequestResult:
    latency_ms: float
    server_infer_ms: float | None
    server_total_ms: float | None
    projected_tokens: Any | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark base VLA policy server /infer endpoint")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--input-profile", choices=INPUT_PROFILE_CHOICES, default="official_demo")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)

    parser.add_argument("--image", default=None, help="Optional RGB image path (png/jpg).")
    parser.add_argument("--goal-image", default=None, help="Optional goal image path (png/jpg).")
    parser.add_argument("--edge-current-image", default=None, help="Optional 2nd ego image path for visualization (png/jpg).")
    parser.add_argument("--send-goal-image-to-vla", action="store_true", help="Send goal_image to the VLA request payload.")
    parser.add_argument("--include-goal-image", action="store_true", help="Send goal image using current image bytes.")
    parser.add_argument("--image-key", default="front_image")
    parser.add_argument("--goal-image-key", default="goal_image")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--asyncvla-repo-dir", default="~/gitrepo/AsyncVLA")
    parser.add_argument("--hf-dir", default="~/huggingface/AsyncVLA_release")
    parser.add_argument("--save-visualization", default="artifacts/benchmark_base_vla_server_result.png")
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=-10.0)
    parser.add_argument("--goal-yaw", type=float, default=-90.0, help="Goal yaw in degrees for official_demo profile.")
    parser.add_argument("--task-mode", choices=TASK_MODE_CHOICES, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--satellite", dest="satellite", action="store_true")
    parser.add_argument("--no-satellite", dest="satellite", action="store_false")
    parser.set_defaults(satellite=None)

    parser.add_argument("--check-health", dest="check_health", action="store_true")
    parser.add_argument("--no-check-health", dest="check_health", action="store_false")
    parser.set_defaults(check_health=True)
    parser.add_argument("--max-errors", type=int, default=5, help="Max failed request messages to print.")
    return parser.parse_args()


def _healthz_url(policy_url: str) -> str:
    parts = urlsplit(policy_url)
    path = parts.path or ""
    if path.endswith("/infer"):
        healthz_path = f"{path[:-6]}/healthz"
    else:
        healthz_path = "/healthz"
    return urlunsplit((parts.scheme, parts.netloc, healthz_path, "", ""))


def _resolve_default_sample_paths(args: argparse.Namespace) -> None:
    inference_dir = Path(args.asyncvla_repo_dir).expanduser().resolve() / "inference"
    if args.image is None:
        args.image = str(inference_dir / "past.png")
    if args.goal_image is None:
        args.goal_image = str(inference_dir / "goal.png")
    if args.edge_current_image is None:
        args.edge_current_image = str(inference_dir / "cur.png")


def _build_goal_pose_payload(args: argparse.Namespace) -> list[float]:
    if args.input_profile == "official_demo":
        yaw_rad = math.radians(float(args.goal_yaw))
        spacing = float(args.metric_waypoint_spacing)
        return [
            float(args.goal_x) / spacing,
            float(args.goal_y) / spacing,
            math.cos(yaw_rad),
            math.sin(yaw_rad),
        ]
    return [float(args.goal_x), float(args.goal_y), float(args.goal_yaw)]


def _should_send_goal_image(args: argparse.Namespace) -> bool:
    if args.include_goal_image:
        return True
    if args.send_goal_image_to_vla:
        return True
    if args.input_profile == "official_demo":
        return args.goal_image is not None
    return args.goal_image is not None


def _resolve_runtime_defaults(args: argparse.Namespace) -> None:
    if args.input_profile == "official_demo":
        if args.task_mode is None and args.task_id is None:
            args.task_mode = "pose_only"
        if args.instruction is not None and not str(args.instruction).strip():
            args.instruction = None


def _load_image(path: str | None, width: int, height: int, seed: int) -> tuple[np.ndarray, str]:
    if path is None:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        return image, f"synthetic_random(seed={seed}, shape={image.shape})"

    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    return image, str(image_path)


def _load_pil(path: str) -> Image.Image:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, Image.BILINEAR)
    canvas = Image.new("RGB", size, color=(255, 255, 255))
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def _encode_jpeg_base64(image_rgb: np.ndarray, quality: int) -> str:
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    with BytesIO() as buf:
        img.save(buf, format="JPEG", quality=int(quality))
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_image_blob(image_rgb: np.ndarray, quality: int) -> dict[str, Any]:
    return {
        "encoding": "jpeg_base64",
        "data": _encode_jpeg_base64(image_rgb, quality=quality),
        "shape": list(image_rgb.shape),
    }


def _build_payload_template(
    args: argparse.Namespace,
    image_rgb: np.ndarray,
    image_blob: dict[str, Any],
    goal_blob: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_policy_payload(
        image=image_rgb,
        encoded_image=str(image_blob["data"]),
        current_pose=[0.0, 0.0, 0.0],
        goal_pose=_build_goal_pose_payload(args),
        instruction=args.instruction,
        task_mode=args.task_mode,
        task_id=args.task_id,
        satellite=args.satellite,
        image_key=args.image_key,
        metric_waypoint_spacing=float(args.metric_waypoint_spacing),
    )
    if goal_blob is not None:
        payload["images"][args.goal_image_key] = goal_blob
    return payload


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _send_one_request(policy_url: str, timeout_s: float, payload_template: dict[str, Any]) -> RequestResult:
    payload = dict(payload_template)
    payload["timestamp_ns"] = time.monotonic_ns()
    started_at = time.perf_counter()
    try:
        resp = _get_thread_session().post(policy_url, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        if resp.status_code != 200:
            body = resp.text.strip().replace("\n", " ")
            if len(body) > 220:
                body = f"{body[:220]}..."
            return RequestResult(
                latency_ms=latency_ms,
                server_infer_ms=None,
                server_total_ms=None,
                projected_tokens=None,
                error=f"HTTP {resp.status_code}: {body}",
            )

        result = resp.json()
        if "projected_tokens" not in result:
            return RequestResult(
                latency_ms=latency_ms,
                server_infer_ms=None,
                server_total_ms=None,
                projected_tokens=None,
                error="Missing projected_tokens in response body",
            )
        return RequestResult(
            latency_ms=latency_ms,
            server_infer_ms=_to_float_or_none(result.get("server_infer_ms")),
            server_total_ms=_to_float_or_none(result.get("server_total_ms")),
            projected_tokens=result.get("projected_tokens"),
            error=None,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return RequestResult(
            latency_ms=latency_ms,
            server_infer_ms=None,
            server_total_ms=None,
            projected_tokens=None,
            error=str(exc),
        )


def _print_stats(name: str, values: list[float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    print(
        f"{name}: "
        f"mean={float(arr.mean()):.3f} std={float(arr.std()):.3f} "
        f"min={float(arr.min()):.3f} p50={float(np.percentile(arr, 50)):.3f} "
        f"p90={float(np.percentile(arr, 90)):.3f} p95={float(np.percentile(arr, 95)):.3f} "
        f"p99={float(np.percentile(arr, 99)):.3f} max={float(arr.max()):.3f}"
    )


def _prep_edge_image(image_path: str) -> torch.Tensor:
    image = _load_pil(image_path).resize((96, 96), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]
    poses = [torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)]

    for t in range(1, dx.shape[1]):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    return torch.stack(poses, dim=1)


def _save_visualization(
    hf_dir: str,
    past_image_path: str,
    edge_current_image_path: str,
    goal_image_path: str,
    projected_tokens: Any,
    save_path: str,
    goal_pose: list[float],
    metric_waypoint_spacing: float,
) -> Path:
    model, _, _, _ = load_edge_adapter_from_hf_snapshot(hf_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32).eval()

    projected = torch.as_tensor(projected_tokens, dtype=torch.float32, device=device)
    if projected.ndim == 2:
        projected = projected.unsqueeze(0)
    if projected.ndim != 3:
        raise ValueError(f"Unexpected projected_tokens shape: {tuple(projected.shape)}")

    past_tensor = _prep_edge_image(past_image_path).to(device)
    current_tensor = _prep_edge_image(edge_current_image_path).to(device)

    with torch.no_grad():
        predicted_dactions_past = model(obs_img=past_tensor, past_img=past_tensor, vla_feature=projected)
        predicted_dactions_current = model(obs_img=current_tensor, past_img=past_tensor, vla_feature=projected)
        predicted_actions_past = _delta_to_pose(predicted_dactions_past).cpu().numpy()
        predicted_actions_current = _delta_to_pose(predicted_dactions_current).cpu().numpy()

    predicted_actions_past[..., :2] *= float(metric_waypoint_spacing)
    predicted_actions_current[..., :2] *= float(metric_waypoint_spacing)

    past_img = _load_pil(past_image_path)
    current_img = _load_pil(edge_current_image_path)
    goal_img = _load_pil(goal_image_path)
    output_path = Path(save_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    margin = 24
    image_cell = (480, 360)
    plot_cell = (720, 520)
    canvas = Image.new(
        "RGB",
        (margin * 4 + image_cell[0] * 3, margin * 3 + image_cell[1] + plot_cell[1]),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    panels = [
        (_fit_image(past_img, image_cell), (margin, margin), "VLA input image (past.png)"),
        (_fit_image(current_img, image_cell), (margin * 2 + image_cell[0], margin), "Edge adapter current image (cur.png)"),
        (_fit_image(goal_img, image_cell), (margin * 3 + image_cell[0] * 2, margin), "Goal image (goal.png)"),
    ]
    for panel_img, offset, title in panels:
        canvas.paste(panel_img, offset)
        draw.text((offset[0] + 16, offset[1] + 16), title, fill=(0, 0, 0))

    traj_specs = [
        (predicted_actions_past, (0, 102, 204), "edge obs = past.png"),
        (predicted_actions_current, (204, 0, 0), "edge obs = cur.png"),
    ]

    def _trajectory_points(actions: np.ndarray) -> list[tuple[float, float]]:
        x_seq = actions[0, :, 0]
        y_seq_inv = -actions[0, :, 1]
        return [(0.0, 0.0)] + list(zip(y_seq_inv.tolist(), x_seq.tolist()))

    def _draw_plot_panel(
        panel_offset: tuple[int, int],
        title: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        include_goal: bool,
    ) -> None:
        plot_margin = 48
        plot_box = (
            panel_offset[0] + plot_margin,
            panel_offset[1] + plot_margin,
            panel_offset[0] + plot_cell[0] - plot_margin,
            panel_offset[1] + plot_cell[1] - plot_margin,
        )
        draw.rectangle(
            (panel_offset[0], panel_offset[1], panel_offset[0] + plot_cell[0], panel_offset[1] + plot_cell[1]),
            fill=(255, 255, 255),
        )
        draw.text((panel_offset[0] + 16, panel_offset[1] + 16), title, fill=(0, 0, 0))
        draw.rectangle(plot_box, outline=(90, 90, 90), width=2)

        def to_px(x_coord: float, y_coord: float) -> tuple[int, int]:
            px = plot_box[0] + int(round((x_coord - x_min) / (x_max - x_min) * (plot_box[2] - plot_box[0])))
            py = plot_box[3] - int(round((y_coord - y_min) / (y_max - y_min) * (plot_box[3] - plot_box[1])))
            return px, py

        grid_step_x = max(0.25, round((x_max - x_min) / 6.0, 2))
        grid_step_y = max(0.25, round((y_max - y_min) / 6.0, 2))
        grid_x = np.arange(np.floor(x_min / grid_step_x) * grid_step_x, x_max + grid_step_x, grid_step_x)
        grid_y = np.arange(np.floor(y_min / grid_step_y) * grid_step_y, y_max + grid_step_y, grid_step_y)

        for value in grid_x:
            x0, y0 = to_px(float(value), y_min)
            x1, y1 = to_px(float(value), y_max)
            draw.line((x0, y0, x1, y1), fill=(225, 225, 225), width=1)
        for value in grid_y:
            x0, y0 = to_px(x_min, float(value))
            x1, y1 = to_px(x_max, float(value))
            draw.line((x0, y0, x1, y1), fill=(225, 225, 225), width=1)

        if x_min <= 0.0 <= x_max:
            x0, y0 = to_px(0.0, y_min)
            x1, y1 = to_px(0.0, y_max)
            draw.line((x0, y0, x1, y1), fill=(120, 120, 120), width=2)
        if y_min <= 0.0 <= y_max:
            x0, y0 = to_px(x_min, 0.0)
            x1, y1 = to_px(x_max, 0.0)
            draw.line((x0, y0, x1, y1), fill=(120, 120, 120), width=2)

        legend_y = panel_offset[1] + 48
        for actions, color, label in traj_specs:
            pixel_points = [to_px(x_coord, y_coord) for x_coord, y_coord in _trajectory_points(actions)]
            draw.line(pixel_points, fill=color, width=4)
            for point in pixel_points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
            draw.rectangle((panel_offset[0] + 16, legend_y - 8, panel_offset[0] + 36, legend_y + 8), fill=color)
            draw.text((panel_offset[0] + 44, legend_y - 10), label, fill=(0, 0, 0))
            legend_y += 28

        if include_goal:
            goal_px = to_px(-goal_pose[1], goal_pose[0])
            draw.line((goal_px[0] - 8, goal_px[1], goal_px[0] + 8, goal_px[1]), fill=(0, 0, 0), width=3)
            draw.line((goal_px[0], goal_px[1] - 8, goal_px[0], goal_px[1] + 8), fill=(0, 0, 0), width=3)
            draw.text((goal_px[0] + 10, goal_px[1] - 10), "goal pose", fill=(0, 0, 0))

        draw.text(
            (panel_offset[0] + 16, panel_offset[1] + plot_cell[1] - 28),
            f"x=[{x_min:.2f}, {x_max:.2f}]m  y=[{y_min:.2f}, {y_max:.2f}]m",
            fill=(0, 0, 0),
        )

    overview_xy: list[tuple[float, float]] = [(0.0, 0.0), (-float(goal_pose[1]), float(goal_pose[0]))]
    zoom_xy: list[tuple[float, float]] = [(0.0, 0.0)]
    for actions, _, _ in traj_specs:
        points = _trajectory_points(actions)
        overview_xy.extend(points[1:])
        zoom_xy.extend(points[1:])

    def _bounds(points: list[tuple[float, float]], min_span: float, include_floor_zero: bool) -> tuple[float, float, float, float]:
        xs = np.asarray([xy[0] for xy in points], dtype=np.float32)
        ys = np.asarray([xy[1] for xy in points], dtype=np.float32)
        x_span = max(float(xs.max() - xs.min()), min_span)
        y_span = max(float(ys.max() - ys.min()), min_span)
        bound_margin = 0.15
        x_min = float(xs.min() - x_span * bound_margin)
        x_max = float(xs.max() + x_span * bound_margin)
        y_min = float(ys.min() - y_span * bound_margin)
        y_max = float(ys.max() + y_span * bound_margin)
        if include_floor_zero:
            y_min = min(-0.05, y_min)
        if x_max - x_min < 1e-6:
            x_min -= min_span * 0.5
            x_max += min_span * 0.5
        if y_max - y_min < 1e-6:
            y_min -= min_span * 0.5
            y_max += min_span * 0.5
        return x_min, x_max, y_min, y_max

    overview_bounds = _bounds(overview_xy, min_span=0.5, include_floor_zero=True)
    zoom_bounds = _bounds(zoom_xy, min_span=0.2, include_floor_zero=True)

    overview_offset = (margin, margin * 2 + image_cell[1])
    zoom_offset = (margin * 2 + plot_cell[0], margin * 2 + image_cell[1])
    _draw_plot_panel(overview_offset, "Predicted trajectories (meters, overview)", *overview_bounds, include_goal=True)
    _draw_plot_panel(zoom_offset, "Predicted trajectories (meters, zoomed)", *zoom_bounds, include_goal=False)

    canvas.save(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be > 0")

    _resolve_default_sample_paths(args)
    _resolve_runtime_defaults(args)

    if args.check_health:
        healthz_url = _healthz_url(args.policy_url)
        healthz_resp = requests.get(healthz_url, timeout=args.timeout_s)
        healthz_resp.raise_for_status()
        print(f"healthz: ok ({healthz_url})")

    image_rgb, image_source = _load_image(args.image, args.image_width, args.image_height, args.image_seed)
    image_blob = _make_image_blob(image_rgb, quality=args.jpeg_quality)

    goal_blob: dict[str, Any] | None = None
    if _should_send_goal_image(args) and args.goal_image is not None:
        goal_rgb, _ = _load_image(args.goal_image, args.image_width, args.image_height, args.image_seed)
        goal_blob = _make_image_blob(goal_rgb, quality=args.jpeg_quality)
    elif args.include_goal_image:
        goal_blob = dict(image_blob)

    payload_template = _build_payload_template(
        args,
        image_rgb=image_rgb,
        image_blob=image_blob,
        goal_blob=goal_blob,
    )

    print(f"policy_url: {args.policy_url}")
    print(f"input_profile: {args.input_profile}")
    print(f"image: {image_source}")
    print(f"image_key: {args.image_key} shape={tuple(image_rgb.shape)}")
    goal_pose_payload = payload_template.get("goal_pose")
    if goal_pose_payload is not None:
        print(f"goal_pose_payload: {goal_pose_payload}")
    else:
        print("goal_pose_payload: disabled_for_task_mode")
    if args.task_mode is not None:
        print(f"task_mode: {args.task_mode}")
    if args.instruction is not None:
        print(f"instruction: {args.instruction}")
    if args.goal_image is not None:
        print(f"goal_image: {Path(args.goal_image).expanduser().resolve()}")
    if goal_blob is not None:
        print(f"goal_image_key: {args.goal_image_key} (sent_to_vla)")
    else:
        print("goal_image_key: disabled_for_vla")
    print(f"edge_current_image: {Path(args.edge_current_image).expanduser().resolve()}")
    print(
        "benchmark: "
        f"warmup={args.warmup} runs={args.runs} concurrency={args.concurrency} timeout_s={args.timeout_s}"
    )

    for i in range(args.warmup):
        warmup_result = _send_one_request(args.policy_url, args.timeout_s, payload_template)
        if warmup_result.error is not None:
            raise RuntimeError(f"Warmup request failed at iter={i + 1}: {warmup_result.error}")

    started_at = time.perf_counter()
    results: list[RequestResult] = []
    if args.concurrency == 1:
        for _ in range(args.runs):
            results.append(_send_one_request(args.policy_url, args.timeout_s, payload_template))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_send_one_request, args.policy_url, args.timeout_s, payload_template)
                for _ in range(args.runs)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    elapsed_s = time.perf_counter() - started_at

    errors = [r.error for r in results if r.error is not None]
    success = [r for r in results if r.error is None]

    print(f"requests: attempted={len(results)} success={len(success)} failed={len(errors)}")
    print(f"throughput_rps: attempted={len(results) / elapsed_s:.2f} success={len(success) / elapsed_s:.2f}")

    if success:
        _print_stats("latency_ms(e2e)", [r.latency_ms for r in success])

        server_infer_values = [v for v in (r.server_infer_ms for r in success) if v is not None]
        if server_infer_values:
            _print_stats("server_infer_ms", server_infer_values)

        server_total_values = [v for v in (r.server_total_ms for r in success) if v is not None]
        if server_total_values:
            _print_stats("server_total_ms", server_total_values)
            overhead = [
                r.latency_ms - r.server_total_ms
                for r in success
                if r.server_total_ms is not None
            ]
            if overhead:
                _print_stats("client+network_overhead_ms", overhead)

    if errors:
        print("sample_errors:")
        for msg in errors[: args.max_errors]:
            print(f"  - {msg}")
        return 1
    if not success:
        return 1
    vis_source = next((r.projected_tokens for r in success if r.projected_tokens is not None), None)
    if vis_source is None:
        raise RuntimeError("No projected_tokens available for visualization")
    vis_path = _save_visualization(
        hf_dir=args.hf_dir,
        past_image_path=args.image,
        edge_current_image_path=args.edge_current_image,
        goal_image_path=args.goal_image,
        projected_tokens=vis_source,
        save_path=args.save_visualization,
        goal_pose=[float(args.goal_x), float(args.goal_y), float(args.goal_yaw)],
        metric_waypoint_spacing=float(args.metric_waypoint_spacing),
    )
    print(f"saved_visualization: {vis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
