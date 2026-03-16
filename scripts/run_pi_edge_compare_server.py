#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
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

from asyncvla_pi import HailoEdgeRunner, HailoEdgeRunnerConfig, TorchEdgeRunner, TorchEdgeRunnerConfig


def _decode_image_blob(blob: Any) -> np.ndarray:
    if isinstance(blob, dict):
        encoding = str(blob.get("encoding", "")).strip().lower()
        if encoding == "jpeg_base64":
            raw = base64.b64decode(blob["data"])
            image = Image.open(BytesIO(raw)).convert("RGB")
            return np.asarray(image, dtype=np.uint8)
        if "data" in blob and "shape" in blob:
            arr = np.asarray(blob["data"], dtype=np.uint8)
            return arr.reshape(blob["shape"])  # type: ignore[arg-type]
    arr = np.asarray(blob, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image payload shape: {arr.shape}")
    return arr


def _decode_array_blob(blob: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(blob, dict):
        encoding = str(blob.get("encoding", "")).strip().lower()
        if encoding == "npy_base64":
            raw = base64.b64decode(blob["data"])
            with BytesIO(raw) as buffer:
                return np.asarray(np.load(buffer, allow_pickle=False), dtype=dtype)
        if "data" in blob and "shape" in blob:
            arr = np.asarray(blob["data"], dtype=dtype)
            return arr.reshape(blob["shape"])  # type: ignore[arg-type]
    return np.asarray(blob, dtype=dtype)


def _extract_image(payload: dict[str, Any], key: str) -> np.ndarray | None:
    if key in payload:
        return _decode_image_blob(payload[key])
    images = payload.get("images")
    if isinstance(images, dict) and key in images:
        return _decode_image_blob(images[key])
    return None


def _extract_required_image(payload: dict[str, Any], key: str) -> np.ndarray:
    image = _extract_image(payload, key)
    if image is None:
        raise KeyError(f"Missing '{key}' image in request payload")
    return image


def _extract_projected_tokens(payload: dict[str, Any]) -> np.ndarray:
    if "projected_tokens" not in payload:
        raise KeyError("Missing 'projected_tokens' in request payload")
    tokens = _decode_array_blob(payload["projected_tokens"], dtype=np.float32)
    if tokens.ndim not in {2, 3}:
        raise ValueError(f"Unsupported projected_tokens shape: {tokens.shape}")
    return np.asarray(tokens, dtype=np.float32)


def _extract_goal_pose(payload: dict[str, Any]) -> np.ndarray | None:
    if "goal_pose" not in payload or payload["goal_pose"] is None:
        return None
    goal = _decode_array_blob(payload["goal_pose"], dtype=np.float32).reshape(-1)
    if goal.size == 0:
        return None
    return goal


def _compare_outputs(
    torch_out: np.ndarray,
    hailo_out: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    torch_arr = np.asarray(torch_out, dtype=np.float32)
    hailo_arr = np.asarray(hailo_out, dtype=np.float32)
    if torch_arr.shape != hailo_arr.shape:
        return {
            "shape_match": False,
            "torch_shape": list(torch_arr.shape),
            "hef_shape": list(hailo_arr.shape),
        }

    diff = torch_arr - hailo_arr
    torch_flat = torch_arr.reshape(-1)
    hailo_flat = hailo_arr.reshape(-1)
    denom = float(np.linalg.norm(torch_flat) * np.linalg.norm(hailo_flat))
    cosine = float(np.dot(torch_flat, hailo_flat) / denom) if denom > 0.0 else 1.0
    return {
        "shape_match": True,
        "shape": list(torch_arr.shape),
        "max_abs": float(np.abs(diff).max()),
        "mean_abs": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_signed": float(diff.mean()),
        "cosine_similarity": cosine,
        "allclose": bool(np.allclose(torch_arr, hailo_arr, rtol=rtol, atol=atol)),
        "allclose_rtol": float(rtol),
        "allclose_atol": float(atol),
    }


class EdgeCompareService:
    def __init__(self, args: argparse.Namespace) -> None:
        if TorchEdgeRunner is None or TorchEdgeRunnerConfig is None:
            raise RuntimeError("TorchEdgeRunner is unavailable in this Python environment")

        self.lock = threading.Lock()
        self.duplicate_current_as_delayed = bool(args.duplicate_current_as_delayed)
        self.allclose_rtol = float(args.allclose_rtol)
        self.allclose_atol = float(args.allclose_atol)
        self.uploaded_hef_dir = Path(args.uploaded_hef_dir).expanduser().resolve()
        self.uploaded_hef_dir.mkdir(parents=True, exist_ok=True)
        self.active_hef_path: Path | None = None
        self.hailo_runner: HailoEdgeRunner | None = None
        self._hailo_runner_kwargs = {
            "input_current_image": args.input_current_name,
            "input_delayed_image": args.input_delayed_name,
            "input_projected_tokens": args.input_tokens_name,
            "input_goal_pose": args.input_goal_name,
            "output_action_chunk": args.output_chunk_name,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "chunk_size": None,
            "pose_dim": None,
            "normalize_imagenet": args.normalize_imagenet,
            "image_layout": args.image_layout,
            "input_format_type": args.input_format,
            "output_format_type": args.output_format,
            "image_scale_255": args.image_scale_255,
            "convert_bgr_to_rgb": args.convert_bgr_to_rgb,
        }

        self.torch_runner = TorchEdgeRunner(
            TorchEdgeRunnerConfig(
                hf_dir=args.hf_dir,
                checkpoint_name=args.shead_checkpoint,
                mha_num_attention_heads=args.mha_num_attention_heads,
                image_height=args.image_height,
                image_width=args.image_width,
                normalize_imagenet=args.normalize_imagenet,
                image_scale_255=args.image_scale_255,
                convert_bgr_to_rgb=args.convert_bgr_to_rgb,
                device=args.torch_device,
                dtype=args.torch_dtype,
            )
        )
        self._hailo_runner_kwargs["chunk_size"] = int(self.torch_runner.model.action_chunk_size)
        self._hailo_runner_kwargs["pose_dim"] = int(self.torch_runner.model.action_dim)
        if args.hef:
            self.set_hef(Path(args.hef).expanduser().resolve())

    def _create_hailo_runner(self, hef_path: Path) -> HailoEdgeRunner:
        return HailoEdgeRunner(
            HailoEdgeRunnerConfig(
                hef_path=str(hef_path),
                input_current_image=str(self._hailo_runner_kwargs["input_current_image"]),
                input_delayed_image=str(self._hailo_runner_kwargs["input_delayed_image"]),
                input_projected_tokens=str(self._hailo_runner_kwargs["input_projected_tokens"]),
                input_goal_pose=self._hailo_runner_kwargs["input_goal_pose"],
                output_action_chunk=str(self._hailo_runner_kwargs["output_action_chunk"]),
                image_height=int(self._hailo_runner_kwargs["image_height"]),
                image_width=int(self._hailo_runner_kwargs["image_width"]),
                chunk_size=int(self._hailo_runner_kwargs["chunk_size"]),
                pose_dim=int(self._hailo_runner_kwargs["pose_dim"]),
                normalize_imagenet=bool(self._hailo_runner_kwargs["normalize_imagenet"]),
                image_layout=str(self._hailo_runner_kwargs["image_layout"]),
                input_format_type=str(self._hailo_runner_kwargs["input_format_type"]),
                output_format_type=str(self._hailo_runner_kwargs["output_format_type"]),
                image_scale_255=bool(self._hailo_runner_kwargs["image_scale_255"]),
                convert_bgr_to_rgb=bool(self._hailo_runner_kwargs["convert_bgr_to_rgb"]),
            )
        )

    def set_hef(self, hef_path: Path) -> dict[str, Any]:
        resolved = hef_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"HEF not found: {resolved}")
        with self.lock:
            if self.hailo_runner is not None:
                self.hailo_runner.close()
            self.hailo_runner = self._create_hailo_runner(resolved)
            self.active_hef_path = resolved
        return {"hef_path": str(resolved)}

    def upload_hef_bytes(self, filename: str, raw: bytes) -> dict[str, Any]:
        if not raw:
            raise ValueError("Uploaded HEF payload is empty")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        safe_name = Path(filename or "uploaded.hef").name
        stored_path = self.uploaded_hef_dir / f"{digest}_{safe_name}"
        if not stored_path.exists():
            stored_path.write_bytes(raw)
        result = self.set_hef(stored_path)
        result["uploaded"] = True
        result["sha256_prefix"] = digest
        return result

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_image = _extract_required_image(payload, "current_image")
        delayed_image = _extract_image(payload, "delayed_image")
        if delayed_image is None:
            if not self.duplicate_current_as_delayed:
                raise KeyError("Missing 'delayed_image' in request payload")
            delayed_image = current_image

        projected_tokens = _extract_projected_tokens(payload)
        goal_pose = _extract_goal_pose(payload)
        timestamp_ns = int(payload.get("timestamp_ns", time.monotonic_ns()))

        with self.lock:
            t0 = time.perf_counter()

            t_torch = time.perf_counter()
            torch_out = self.torch_runner.infer(
                current_image=current_image,
                delayed_image=delayed_image,
                projected_tokens=projected_tokens,
                goal_pose=goal_pose,
            )
            torch_ms = (time.perf_counter() - t_torch) * 1000.0

            t_hailo = time.perf_counter()
            if self.hailo_runner is None:
                raise RuntimeError("No active HEF is loaded. Upload one with POST /hef first.")
            hailo_out = self.hailo_runner.infer(
                current_image=current_image,
                delayed_image=delayed_image,
                projected_tokens=projected_tokens,
                goal_pose=goal_pose,
            )
            hailo_ms = (time.perf_counter() - t_hailo) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0

        torch_arr = np.asarray(torch_out, dtype=np.float32)
        hailo_arr = np.asarray(hailo_out, dtype=np.float32)
        return {
            "ok": True,
            "timestamp_ns": timestamp_ns,
            "torch_action_chunk": torch_arr.tolist(),
            "hef_action_chunk": hailo_arr.tolist(),
            "diff_report": _compare_outputs(
                torch_arr,
                hailo_arr,
                rtol=self.allclose_rtol,
                atol=self.allclose_atol,
            ),
            "latency_ms": {
                "torch": torch_ms,
                "hef": hailo_ms,
                "total": total_ms,
            },
            "input_summary": {
                "current_image_shape": list(current_image.shape),
                "delayed_image_shape": list(delayed_image.shape),
                "projected_tokens_shape": list(projected_tokens.shape),
                "goal_pose_shape": None if goal_pose is None else list(goal_pose.shape),
            },
            "hef_path": None if self.active_hef_path is None else str(self.active_hef_path),
        }

    def close(self) -> None:
        if self.hailo_runner is not None:
            self.hailo_runner.close()
        self.torch_runner.close()


class CompareHandler(BaseHTTPRequestHandler):
    service: EdgeCompareService

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
            self._send_json(
                200,
                {
                    "ok": True,
                    "hef_path": None if self.service.active_hef_path is None else str(self.service.active_hef_path),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/hef":
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len)
                filename = self.headers.get("X-Filename", "uploaded.hef")
                response = self.service.upload_hef_bytes(filename, raw)
                self._send_json(200, {"ok": True, **response})
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return

        if self.path != "/infer":
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("Request body must decode to a JSON object")
            response = self.service.infer(payload)
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Pi-side edge adapter server that returns both HEF and Torch outputs for the same input."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--hef", default="models/edge_adapter_v520.hef")
    parser.add_argument("--shead-checkpoint", default="shead--750000_checkpoint.pt")
    parser.add_argument("--mha-num-attention-heads", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--normalize-imagenet", action="store_true")
    parser.add_argument("--image-scale-255", action="store_true")
    parser.add_argument("--input-current-name", default="edge/input_layer1")
    parser.add_argument("--input-delayed-name", default="edge/input_layer2")
    parser.add_argument("--input-tokens-name", default="edge/input_layer3")
    parser.add_argument("--input-goal-name", default=None)
    parser.add_argument("--output-chunk-name", default="edge/depth_to_space1")
    parser.add_argument("--image-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--input-format", choices=["uint8", "float32", "auto"], default="uint8")
    parser.add_argument("--output-format", choices=["uint8", "float32", "auto"], default="auto")
    parser.add_argument(
        "--duplicate-current-as-delayed",
        action="store_true",
        help="If delayed_image is omitted in the request, reuse current_image.",
    )
    parser.add_argument("--allclose-rtol", type=float, default=1e-2)
    parser.add_argument("--allclose-atol", type=float, default=1e-2)
    parser.add_argument("--uploaded-hef-dir", default="/tmp/asyncvla_uploaded_hef")
    parser.add_argument(
        "--allow-missing-initial-hef",
        action="store_true",
        help="Start server even if --hef does not exist yet; upload one later via POST /hef.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_dir = Path(args.hf_dir).expanduser().resolve()
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF directory not found: {hf_dir}")

    hef_path = Path(args.hef).expanduser().resolve() if args.hef else None
    if hef_path is not None and not hef_path.exists():
        if args.allow_missing_initial_hef:
            hef_path = None
        else:
            raise FileNotFoundError(f"HEF not found: {hef_path}")
    args.hef = "" if hef_path is None else str(hef_path)

    service = EdgeCompareService(args)
    handler = CompareHandler
    handler.service = service

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Edge compare server listening: http://{args.host}:{args.port}/infer")
    print(f"HF snapshot: {hf_dir}")
    print(f"HEF: {hef_path if hef_path is not None else '(none; waiting for upload via POST /hef)'}")
    print(f"Torch device/dtype: {args.torch_device}/{args.torch_dtype}")
    print(
        "Preprocess: "
        f"convert_bgr_to_rgb={args.convert_bgr_to_rgb} "
        f"normalize_imagenet={args.normalize_imagenet} "
        f"image_scale_255={args.image_scale_255}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
