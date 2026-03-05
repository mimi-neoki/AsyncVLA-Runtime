#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot_policy_asyncvla_base import AsyncVLABasePolicy


def _decode_image_blob(blob: Any) -> np.ndarray:
    if isinstance(blob, dict):
        encoding = str(blob.get("encoding", "")).lower()
        if encoding == "jpeg_base64":
            raw = base64.b64decode(blob["data"])
            img = Image.open(BytesIO(raw)).convert("RGB")
            return np.asarray(img)
        if "data" in blob and "shape" in blob:
            arr = np.asarray(blob["data"], dtype=np.uint8)
            return arr.reshape(blob["shape"])  # type: ignore[arg-type]
    arr = np.asarray(blob)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image payload shape: {arr.shape}")
    return arr


def _prepare_observation(payload: dict[str, Any], image_key: str) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "timestamp_ns": int(payload.get("timestamp_ns", time.monotonic_ns())),
        "instruction": payload.get("instruction", ""),
        "goal_pose": payload.get("goal_pose", [0.0, 0.0, 0.0]),
        "task_mode": payload.get("task_mode"),
        "task_id": payload.get("task_id"),
        "satellite": bool(payload.get("satellite", False)),
    }

    images = payload.get("images", {})
    if not isinstance(images, dict) or image_key not in images:
        raise KeyError(f"Missing images.{image_key} in request payload")

    decoded_images: dict[str, np.ndarray] = {}
    for key, blob in images.items():
        decoded_images[key] = _decode_image_blob(blob)
    obs[image_key] = decoded_images[image_key]
    obs["images"] = decoded_images
    return obs


class PolicyHandler(BaseHTTPRequestHandler):
    policy: AsyncVLABasePolicy
    image_key: str

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/healthz"}:
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            payload = json.loads(raw.decode("utf-8"))
            observation = _prepare_observation(payload, image_key=self.image_key)
            result = self.policy.select_action(observation)
            response = {
                "projected_tokens": np.asarray(result["projected_tokens"], dtype=np.float32).tolist(),
                "timestamp_ns": int(result["timestamp_ns"]),
            }
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base-VLA guidance server for Pi edge adapter client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--hf-dir",
        default="~/gitrepo/AsyncVLA_release",
        help="HF snapshot containing base model + projector checkpoints",
    )
    parser.add_argument(
        "--asyncvla-repo-dir",
        default="~/gitrepo/AsyncVLA",
        help="Path to official AsyncVLA repo (for prismatic namespace during HF trust_remote_code import).",
    )
    parser.add_argument("--image-key", default="front_image")
    parser.add_argument("--goal-image-key", default="goal_image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", choices=["none", "8bit"], default="none")
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
        default="auto",
        help="Task mode mapped to AsyncVLA modality_id (0-8).",
    )
    parser.add_argument(
        "--satellite-default",
        action="store_true",
        help="Default satellite=True for auto task-mode resolution when request does not set satellite.",
    )
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--disable-goal-image-duplication",
        action="store_true",
        help="If set, require images.goal_image in request payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_dir = Path(args.hf_dir).expanduser().resolve()
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF directory not found: {hf_dir}")

    policy = AsyncVLABasePolicy.from_snapshot(
        snapshot_dir=str(hf_dir),
        image_key=args.image_key,
        goal_image_key=args.goal_image_key,
        asyncvla_repo_dir=args.asyncvla_repo_dir,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        task_mode=args.task_mode,
        satellite_default=args.satellite_default,
        num_images_in_input=args.num_images_in_input,
        unnorm_key=args.unnorm_key,
        task_id=args.task_id,
        duplicate_current_image_as_goal=not args.disable_goal_image_duplication,
    )

    handler = PolicyHandler
    handler.policy = policy
    handler.image_key = args.image_key

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Base guidance server listening: http://{args.host}:{args.port}/infer")
    print(f"HF snapshot: {hf_dir}")
    print(f"Image key: {args.image_key}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
