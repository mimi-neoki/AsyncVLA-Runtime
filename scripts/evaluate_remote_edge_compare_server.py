#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot_policy_asyncvla_base import AsyncVLABasePolicy


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


@dataclass
class RequestResult:
    roundtrip_ms: float
    response: dict[str, Any] | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server-side evaluator that generates edge inputs and sends them to the Pi edge compare server."
    )
    parser.add_argument("--edge-url", default="http://127.0.0.1:8100/infer")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--check-health", dest="check_health", action="store_true")
    parser.add_argument("--no-check-health", dest="check_health", action="store_false")
    parser.set_defaults(check_health=True)

    parser.add_argument("--projected-tokens", default=None, help="Optional .npy path. If set, skip base policy.")
    parser.add_argument("--upload-hef", default="", help="Optional HEF path to upload to the Pi before evaluation.")
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--asyncvla-repo-dir", default="~/gitrepo/AsyncVLA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", choices=["none", "8bit"], default="none")
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--unnorm-key", default=None)

    parser.add_argument("--image", default=None, help="Base policy current image.")
    parser.add_argument("--goal-image", default=None, help="Base policy goal image.")
    parser.add_argument("--edge-current-image", default=None, help="Edge adapter current image.")
    parser.add_argument("--edge-delayed-image", default=None, help="Edge adapter delayed image.")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--image-key", default="front_image")
    parser.add_argument("--goal-image-key", default="goal_image")
    parser.add_argument("--include-goal-image", action="store_true")

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=-10.0)
    parser.add_argument("--goal-yaw", type=float, default=-90.0, help="Degrees.")
    parser.add_argument("--task-mode", choices=TASK_MODE_CHOICES, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--satellite", dest="satellite", action="store_true")
    parser.add_argument("--no-satellite", dest="satellite", action="store_false")
    parser.set_defaults(satellite=None)
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)

    parser.add_argument("--save-json", default="", help="Optional path to save the last response and summary.")
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


def _resolve_default_sample_paths(args: argparse.Namespace) -> None:
    inference_dir = Path(args.asyncvla_repo_dir).expanduser().resolve() / "inference"
    if args.image is None:
        args.image = str(inference_dir / "past.png")
    if args.goal_image is None:
        args.goal_image = str(inference_dir / "goal.png")
    if args.edge_current_image is None:
        args.edge_current_image = str(inference_dir / "cur.png")
    if args.edge_delayed_image is None:
        args.edge_delayed_image = str(inference_dir / "past.png")


def _load_image(path: str | None, width: int, height: int, seed: int) -> tuple[np.ndarray, str]:
    if path is None:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        return image, f"synthetic_random(seed={seed}, shape={image.shape})"

    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if Image is not None:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    elif cv2 is not None:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to load image with OpenCV: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError("Neither PIL nor OpenCV is available for image loading")
    return image, str(image_path)


def _encode_jpeg_base64(image_rgb: np.ndarray, quality: int) -> str:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    if Image is not None:
        img = Image.fromarray(image_rgb, mode="RGB")
        with BytesIO() as buf:
            img.save(buf, format="JPEG", quality=int(quality))
            return base64.b64encode(buf.getvalue()).decode("ascii")
    if cv2 is not None:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("Failed to encode image as JPEG with OpenCV")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    raise RuntimeError("Neither PIL nor OpenCV is available for JPEG encoding")


def _make_image_blob(image_rgb: np.ndarray, quality: int) -> dict[str, Any]:
    return {
        "encoding": "jpeg_base64",
        "data": _encode_jpeg_base64(image_rgb, quality=quality),
        "shape": list(image_rgb.shape),
    }


def _build_goal_pose(args: argparse.Namespace) -> list[float]:
    return [float(args.goal_x), float(args.goal_y), math.radians(float(args.goal_yaw))]


def _load_projected_tokens(path: str) -> np.ndarray:
    tokens_path = Path(path).expanduser().resolve()
    if not tokens_path.exists():
        raise FileNotFoundError(f"Projected tokens file not found: {tokens_path}")
    tokens = np.load(tokens_path)
    return np.asarray(tokens, dtype=np.float32)


def _build_policy_observation(
    args: argparse.Namespace,
    image_rgb: np.ndarray,
    goal_rgb: np.ndarray | None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "timestamp_ns": time.monotonic_ns(),
        args.image_key: image_rgb,
        "images": {args.image_key: image_rgb},
        "goal_pose": _build_goal_pose(args),
        "task_mode": args.task_mode,
        "task_id": args.task_id,
        "satellite": args.satellite,
    }
    if args.instruction is not None:
        observation["instruction"] = args.instruction
    if args.include_goal_image and goal_rgb is not None:
        observation[args.goal_image_key] = goal_rgb
        observation["images"][args.goal_image_key] = goal_rgb
    return observation


def _generate_projected_tokens(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.projected_tokens:
        tokens = _load_projected_tokens(args.projected_tokens)
        return tokens, {"source": str(Path(args.projected_tokens).expanduser().resolve())}

    if AsyncVLABasePolicy is None:
        raise RuntimeError(
            "AsyncVLABasePolicy is unavailable in this Python environment. "
            "Install the server-side model dependencies or pass --projected-tokens."
        )

    hf_dir = Path(args.hf_dir).expanduser().resolve()
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF directory not found: {hf_dir}")

    image_rgb, image_source = _load_image(args.image, args.image_width, args.image_height, args.image_seed)
    goal_rgb = None
    goal_source = None
    if args.goal_image is not None:
        goal_rgb, goal_source = _load_image(args.goal_image, args.image_width, args.image_height, args.image_seed + 1)

    policy = AsyncVLABasePolicy.from_snapshot(
        snapshot_dir=str(hf_dir),
        image_key=args.image_key,
        goal_image_key=args.goal_image_key,
        asyncvla_repo_dir=args.asyncvla_repo_dir,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        task_mode=args.task_mode or "auto",
        satellite_default=bool(args.satellite) if args.satellite is not None else False,
        num_images_in_input=args.num_images_in_input,
        unnorm_key=args.unnorm_key,
        task_id=args.task_id,
        duplicate_current_image_as_goal=not args.include_goal_image,
    )
    observation = _build_policy_observation(args, image_rgb=image_rgb, goal_rgb=goal_rgb)
    result = policy.select_action(observation)
    tokens = np.asarray(result["projected_tokens"], dtype=np.float32)
    meta = {
        "source": "AsyncVLABasePolicy",
        "hf_dir": str(hf_dir),
        "image": image_source,
        "goal_image": goal_source,
        "task_mode": args.task_mode,
        "task_id": args.task_id,
    }
    return tokens, meta


def _build_remote_payload(
    args: argparse.Namespace,
    projected_tokens: np.ndarray,
) -> tuple[dict[str, Any], dict[str, str]]:
    current_rgb, current_source = _load_image(
        args.edge_current_image, args.image_width, args.image_height, args.image_seed + 2
    )
    delayed_rgb, delayed_source = _load_image(
        args.edge_delayed_image, args.image_width, args.image_height, args.image_seed + 3
    )
    payload = {
        "timestamp_ns": time.monotonic_ns(),
        "current_image": _make_image_blob(current_rgb, quality=args.jpeg_quality),
        "delayed_image": _make_image_blob(delayed_rgb, quality=args.jpeg_quality),
        "projected_tokens": np.asarray(projected_tokens, dtype=np.float32).tolist(),
    }
    return payload, {"edge_current_image": current_source, "edge_delayed_image": delayed_source}


def _send_one_request(edge_url: str, timeout_s: float, payload_template: dict[str, Any]) -> RequestResult:
    payload = dict(payload_template)
    payload["timestamp_ns"] = time.monotonic_ns()
    started_at = time.perf_counter()
    try:
        resp = requests.post(edge_url, json=payload, timeout=timeout_s)
        roundtrip_ms = (time.perf_counter() - started_at) * 1000.0
        if resp.status_code != 200:
            body = resp.text.strip().replace("\n", " ")
            return RequestResult(roundtrip_ms=roundtrip_ms, response=None, error=f"HTTP {resp.status_code}: {body}")
        response = resp.json()
        return RequestResult(roundtrip_ms=roundtrip_ms, response=response, error=None)
    except Exception as exc:
        roundtrip_ms = (time.perf_counter() - started_at) * 1000.0
        return RequestResult(roundtrip_ms=roundtrip_ms, response=None, error=str(exc))


def _upload_hef(edge_url: str, hef_path: str, timeout_s: float) -> dict[str, Any]:
    path = Path(hef_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HEF not found: {path}")
    response = requests.post(
        _hef_url(edge_url),
        data=path.read_bytes(),
        headers={"Content-Type": "application/octet-stream", "X-Filename": path.name},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"HEF upload failed: {payload}")
    return payload


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_results(results: list[RequestResult]) -> dict[str, Any]:
    successes = [r for r in results if r.error is None and r.response is not None]
    errors = [r.error for r in results if r.error is not None]

    roundtrip_ms = [r.roundtrip_ms for r in successes]
    torch_ms = [float(r.response["latency_ms"]["torch"]) for r in successes]  # type: ignore[index]
    hef_ms = [float(r.response["latency_ms"]["hef"]) for r in successes]  # type: ignore[index]
    total_ms = [float(r.response["latency_ms"]["total"]) for r in successes]  # type: ignore[index]

    max_abs_values: list[float] = []
    mean_abs_values: list[float] = []
    rmse_values: list[float] = []
    cosine_values: list[float] = []
    allclose_count = 0
    for result in successes:
        diff = result.response.get("diff_report", {})  # type: ignore[union-attr]
        if bool(diff.get("shape_match", False)):
            max_abs_values.append(float(diff["max_abs"]))
            mean_abs_values.append(float(diff["mean_abs"]))
            rmse_values.append(float(diff["rmse"]))
            cosine_values.append(float(diff["cosine_similarity"]))
            if bool(diff.get("allclose", False)):
                allclose_count += 1

    return {
        "runs": len(results),
        "successes": len(successes),
        "errors": errors,
        "roundtrip_ms_mean": _mean(roundtrip_ms),
        "remote_torch_ms_mean": _mean(torch_ms),
        "remote_hef_ms_mean": _mean(hef_ms),
        "remote_total_ms_mean": _mean(total_ms),
        "diff_max_abs_mean": _mean(max_abs_values),
        "diff_mean_abs_mean": _mean(mean_abs_values),
        "diff_rmse_mean": _mean(rmse_values),
        "diff_cosine_similarity_mean": _mean(cosine_values),
        "allclose_rate": (allclose_count / len(successes)) if successes else None,
    }


def _save_json(path: str, payload: dict[str, Any]) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    _resolve_default_sample_paths(args)

    if args.check_health:
        healthz = requests.get(_healthz_url(args.edge_url), timeout=args.timeout_s)
        healthz.raise_for_status()

    if args.upload_hef:
        upload_result = _upload_hef(args.edge_url, args.upload_hef, args.timeout_s)
        print(f"uploaded_hef: {upload_result.get('hef_path')}")

    projected_tokens, token_meta = _generate_projected_tokens(args)
    payload_template, edge_meta = _build_remote_payload(args, projected_tokens)

    print(f"edge_url: {args.edge_url}")
    print(f"projected_tokens: shape={tuple(np.asarray(projected_tokens).shape)} source={token_meta['source']}")
    print(f"edge_current_image: {edge_meta['edge_current_image']}")
    print(f"edge_delayed_image: {edge_meta['edge_delayed_image']}")

    for _ in range(max(args.warmup, 0)):
        _send_one_request(args.edge_url, args.timeout_s, payload_template)

    results = [_send_one_request(args.edge_url, args.timeout_s, payload_template) for _ in range(max(args.runs, 0))]
    summary = _summarize_results(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    last_success = next((r.response for r in reversed(results) if r.response is not None), None)
    if last_success is not None:
        diff_report = last_success.get("diff_report")
        if diff_report is not None:
            print("last_diff_report:")
            print(json.dumps(diff_report, indent=2, ensure_ascii=False))

    if args.save_json:
        save_payload = {
            "summary": summary,
            "token_meta": token_meta,
            "edge_meta": edge_meta,
            "last_success": last_success,
        }
        out_path = _save_json(args.save_json, save_payload)
        print(f"saved_json: {out_path}")

    return 0 if summary["successes"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
