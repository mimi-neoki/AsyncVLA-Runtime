#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and save model-wise edge adapter outputs from the remote compare server."
    )
    parser.add_argument("--edge-url", default="http://openduck.local:8100/infer")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--check-health", dest="check_health", action="store_true")
    parser.add_argument("--no-check-health", dest="check_health", action="store_false")
    parser.set_defaults(check_health=True)

    parser.add_argument("--samples-json", default="calib_data/samples.json")
    parser.add_argument("--images-dir", default="calib_data/images")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--delayed-mode", choices=["roll_fullset", "same"], default="roll_fullset")
    parser.add_argument("--jpeg-quality", type=int, default=90)

    parser.add_argument("--hef", action="append", default=[], help="HEF path. Can be passed multiple times.")
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--trajectory-mode",
        choices=["delta_to_pose", "raw_xy"],
        default="delta_to_pose",
        help="How to convert action chunks before plotting.",
    )
    return parser.parse_args()


def _healthz_url(edge_url: str) -> str:
    parts = urlsplit(edge_url)
    path = parts.path or ""
    if path.endswith("/infer"):
        healthz_path = f"{path[:-6]}/healthz"
    else:
        healthz_path = "/healthz"
    return urlunsplit((parts.scheme, parts.netloc, healthz_path, "", ""))


def _hef_url(edge_url: str) -> str:
    parts = urlsplit(edge_url)
    path = parts.path or ""
    if path.endswith("/infer"):
        hef_path = f"{path[:-6]}/hef"
    else:
        hef_path = "/hef"
    return urlunsplit((parts.scheme, parts.netloc, hef_path, "", ""))


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


def _upload_hef(edge_url: str, hef_path: Path, timeout_s: float) -> dict[str, Any]:
    response = requests.post(
        _hef_url(edge_url),
        data=hef_path.read_bytes(),
        headers={"Content-Type": "application/octet-stream", "X-Filename": hef_path.name},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"HEF upload failed: {payload}")
    return payload


def _post_infer(edge_url: str, timeout_s: float, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(edge_url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    body = response.json()
    if not body.get("ok", False):
        raise RuntimeError(f"Remote infer failed: {body}")
    return body


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _fit_image(image_rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    bg = Image.new("RGB", size, color=(250, 250, 250))
    fitted = img.copy()
    fitted.thumbnail(size, Image.Resampling.BILINEAR)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    bg.paste(fitted, (x, y))
    return bg


def _delta_chunk_to_pose_chunk(delta_chunk: np.ndarray, metric_waypoint_spacing: float) -> np.ndarray:
    chunk = np.asarray(delta_chunk, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2:
        raise ValueError(f"delta_chunk must be 2D, got shape={chunk.shape}")
    dx = chunk[:, 0] if chunk.shape[1] >= 1 else np.zeros((chunk.shape[0],), dtype=np.float32)
    dy = chunk[:, 1] if chunk.shape[1] >= 2 else np.zeros((chunk.shape[0],), dtype=np.float32)
    if chunk.shape[1] >= 4:
        dtheta = np.arctan2(chunk[:, 3], chunk[:, 2]).astype(np.float32)
    elif chunk.shape[1] >= 3:
        dtheta = chunk[:, 2].astype(np.float32)
    else:
        dtheta = np.zeros((chunk.shape[0],), dtype=np.float32)

    poses = np.zeros((chunk.shape[0], 4), dtype=np.float32)
    x = 0.0
    y = 0.0
    theta = 0.0
    for idx in range(chunk.shape[0]):
        ct = float(np.cos(theta))
        st = float(np.sin(theta))
        x += ct * float(dx[idx]) - st * float(dy[idx])
        y += st * float(dx[idx]) + ct * float(dy[idx])
        theta += float(dtheta[idx])
        poses[idx, 0] = x
        poses[idx, 1] = y
        poses[idx, 2] = float(np.cos(theta))
        poses[idx, 3] = float(np.sin(theta))
    poses[:, :2] *= float(metric_waypoint_spacing)
    return poses


def _raw_xy_chunk(action_chunk: np.ndarray, metric_waypoint_spacing: float) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    out = np.zeros((chunk.shape[0], 4), dtype=np.float32)
    if chunk.shape[1] >= 1:
        out[:, 0] = chunk[:, 0] * float(metric_waypoint_spacing)
    if chunk.shape[1] >= 2:
        out[:, 1] = chunk[:, 1] * float(metric_waypoint_spacing)
    return out


def _trajectory_points(action_chunk: np.ndarray, mode: str, metric_waypoint_spacing: float) -> list[tuple[float, float]]:
    if mode == "delta_to_pose":
        pose_chunk = _delta_chunk_to_pose_chunk(action_chunk, metric_waypoint_spacing)
    else:
        pose_chunk = _raw_xy_chunk(action_chunk, metric_waypoint_spacing)
    points = [(0.0, 0.0)]
    for step in pose_chunk:
        x_forward = float(step[0]) if step.size > 0 else 0.0
        y_left = float(step[1]) if step.size > 1 else 0.0
        points.append((-y_left, x_forward))
    return points


def _bounds(all_points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = np.asarray([p[0] for p in all_points], dtype=np.float32)
    ys = np.asarray([p[1] for p in all_points], dtype=np.float32)
    x_span = max(float(xs.max() - xs.min()), 0.5)
    y_span = max(float(ys.max() - ys.min()), 0.5)
    margin = 0.15
    x_min = float(xs.min() - x_span * margin)
    x_max = float(xs.max() + x_span * margin)
    y_min = float(min(-0.05, ys.min() - y_span * margin))
    y_max = float(ys.max() + y_span * margin)
    if x_max - x_min < 1e-6:
        x_min -= 0.25
        x_max += 0.25
    if y_max - y_min < 1e-6:
        y_min -= 0.25
        y_max += 0.25
    return x_min, x_max, y_min, y_max


def _draw_plot(
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    title: str,
    trajectories: list[dict[str, Any]],
) -> None:
    left, top, right, bottom = panel_box
    draw.rectangle(panel_box, fill=(255, 255, 255), outline=(70, 70, 70), width=2)
    draw.text((left + 16, top + 14), title, fill=(0, 0, 0))

    plot_box = (left + 48, top + 48, right - 32, bottom - 44)
    draw.rectangle(plot_box, outline=(100, 100, 100), width=2)

    all_points = [(0.0, 0.0)]
    for traj in trajectories:
        all_points.extend(traj["points"])
    x_min, x_max, y_min, y_max = _bounds(all_points)

    def to_px(x_coord: float, y_coord: float) -> tuple[int, int]:
        px = plot_box[0] + int(round((x_coord - x_min) / (x_max - x_min) * (plot_box[2] - plot_box[0])))
        py = plot_box[3] - int(round((y_coord - y_min) / (y_max - y_min) * (plot_box[3] - plot_box[1])))
        return px, py

    for frac in np.linspace(0.0, 1.0, 7):
        gx = plot_box[0] + int(round(frac * (plot_box[2] - plot_box[0])))
        gy = plot_box[1] + int(round(frac * (plot_box[3] - plot_box[1])))
        draw.line((gx, plot_box[1], gx, plot_box[3]), fill=(230, 230, 230), width=1)
        draw.line((plot_box[0], gy, plot_box[2], gy), fill=(230, 230, 230), width=1)

    if x_min <= 0.0 <= x_max:
        x0, y0 = to_px(0.0, y_min)
        x1, y1 = to_px(0.0, y_max)
        draw.line((x0, y0, x1, y1), fill=(130, 130, 130), width=2)
    if y_min <= 0.0 <= y_max:
        x0, y0 = to_px(x_min, 0.0)
        x1, y1 = to_px(x_max, 0.0)
        draw.line((x0, y0, x1, y1), fill=(130, 130, 130), width=2)

    legend_y = top + 48
    for traj in trajectories:
        px_points = [to_px(x, y) for x, y in traj["points"]]
        draw.line(px_points, fill=traj["color"], width=4)
        for point in px_points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=traj["color"])
        draw.rectangle((left + 16, legend_y - 6, left + 34, legend_y + 6), fill=traj["color"])
        draw.text((left + 42, legend_y - 10), traj["label"], fill=(0, 0, 0))
        legend_y += 24

    draw.text(
        (left + 16, bottom - 26),
        f"x=[{x_min:.2f}, {x_max:.2f}]m  y=[{y_min:.2f}, {y_max:.2f}]m",
        fill=(0, 0, 0),
    )


def main() -> int:
    args = parse_args()
    if args.metric_waypoint_spacing <= 0.0:
        raise ValueError("--metric-waypoint-spacing must be > 0")
    if not args.hef:
        raise RuntimeError("Pass at least one --hef")

    if args.check_health:
        healthz = requests.get(_healthz_url(args.edge_url), timeout=args.timeout_s)
        healthz.raise_for_status()

    samples = json.loads(Path(args.samples_json).expanduser().resolve().read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir).expanduser().resolve()
    projected_tokens = np.load(Path(args.projected_tokens).expanduser().resolve())

    idx = int(args.sample_index)
    if idx < 0 or idx >= len(samples):
        raise IndexError(f"sample-index out of range: {idx} not in [0, {len(samples) - 1}]")

    sample = samples[idx]
    current_path = images_dir / Path(sample["local_image_path"]).name
    delayed_idx = idx if args.delayed_mode == "same" else (idx - 1) % len(samples)
    delayed_sample = samples[delayed_idx]
    delayed_path = images_dir / Path(delayed_sample["local_image_path"]).name
    current_rgb = _load_rgb_image(current_path)
    delayed_rgb = _load_rgb_image(delayed_path)
    tokens = np.asarray(projected_tokens[idx], dtype=np.float32)

    payload = {
        "timestamp_ns": time.monotonic_ns(),
        "current_image": _make_image_blob(current_rgb, quality=args.jpeg_quality),
        "delayed_image": _make_image_blob(delayed_rgb, quality=args.jpeg_quality),
        "projected_tokens": tokens.tolist(),
    }

    colors = [
        (20, 20, 20),
        (200, 0, 0),
        (0, 110, 220),
        (0, 150, 70),
        (180, 90, 0),
        (140, 0, 160),
    ]

    trajectories: list[dict[str, Any]] = []
    stats_lines: list[str] = []
    torch_points: list[tuple[float, float]] | None = None

    for hef_index, hef in enumerate(args.hef):
        hef_path = Path(hef).expanduser().resolve()
        _upload_hef(args.edge_url, hef_path, args.timeout_s)
        result = _post_infer(args.edge_url, args.timeout_s, payload)

        torch_chunk = np.asarray(result["torch_action_chunk"], dtype=np.float32)
        hef_chunk = np.asarray(result["hef_action_chunk"], dtype=np.float32)
        diff = result["diff_report"]

        if torch_points is None:
            torch_points = _trajectory_points(torch_chunk, args.trajectory_mode, args.metric_waypoint_spacing)
            trajectories.append(
                {
                    "label": "torch",
                    "color": colors[0],
                    "points": torch_points,
                }
            )

        trajectories.append(
            {
                "label": hef_path.stem,
                "color": colors[1 + (hef_index % (len(colors) - 1))],
                "points": _trajectory_points(hef_chunk, args.trajectory_mode, args.metric_waypoint_spacing),
            }
        )
        stats_lines.append(
            f"{hef_path.stem}: rmse={float(diff['rmse']):.3f} "
            f"mean_abs={float(diff['mean_abs']):.3f} "
            f"cos={float(diff['cosine_similarity']):.4f}"
        )

    image_cell = (420, 320)
    margin = 24
    plot_h = 540
    text_h = 170 + max(0, len(stats_lines) - 3) * 22
    width = margin * 3 + image_cell[0] * 2
    height = margin * 4 + image_cell[1] + plot_h + text_h
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    current_panel = _fit_image(current_rgb, image_cell)
    delayed_panel = _fit_image(delayed_rgb, image_cell)
    canvas.paste(current_panel, (margin, margin))
    canvas.paste(delayed_panel, (margin * 2 + image_cell[0], margin))
    draw.text((margin + 12, margin + 12), f"current image: sample {idx}", fill=(0, 0, 0))
    draw.text((margin * 2 + image_cell[0] + 12, margin + 12), f"delayed image: sample {delayed_idx}", fill=(0, 0, 0))

    plot_top = margin * 2 + image_cell[1]
    _draw_plot(
        draw,
        (margin, plot_top, width - margin, plot_top + plot_h),
        f"Model outputs ({args.trajectory_mode}, spacing={args.metric_waypoint_spacing})",
        trajectories,
    )

    info_top = plot_top + plot_h + margin
    draw.rectangle((margin, info_top, width - margin, height - margin), fill=(255, 255, 255), outline=(70, 70, 70), width=2)
    draw.text((margin + 16, info_top + 14), "Sample info", fill=(0, 0, 0))
    info_lines = [
        f"instruction: {sample.get('instruction', '')}",
        f"current_path: {current_path.name}",
        f"delayed_path: {delayed_path.name}",
        f"edge_url: {args.edge_url}",
    ] + stats_lines
    y = info_top + 42
    for line in info_lines:
        draw.text((margin + 16, y), line, fill=(0, 0, 0))
        y += 22

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved_visualization: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
