#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import TorchEdgeRunner, TorchEdgeRunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple HEF candidates against the remote edge compare server using calib_data."
    )
    parser.add_argument("--edge-url", default="http://openduck.local:8100/infer")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--check-health", dest="check_health", action="store_true")
    parser.add_argument("--no-check-health", dest="check_health", action="store_false")
    parser.set_defaults(check_health=True)
    parser.add_argument("--warmup", type=int, default=1)

    parser.add_argument("--samples-json", default="calib_data/samples.json")
    parser.add_argument("--images-dir", default="calib_data/images")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "linspace", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--delayed-mode",
        choices=["roll_fullset", "same"],
        default="roll_fullset",
        help="roll_fullset matches calib_data generation: delayed image is previous sample in the full dataset.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--compare-reference",
        choices=["server", "local_torch"],
        default="server",
        help="How to compute diff_report. 'local_torch' compares remote HEF output against a local TorchEdgeRunner.",
    )
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--torch-preprocess-mode",
        choices=["hf", "hailo_int8norm"],
        default="hf",
    )
    parser.add_argument("--token-quant-mode", choices=["dynamic_minmax", "fixed_affine", "none"], default="dynamic_minmax")
    parser.add_argument("--token-quant-params", default=None)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)

    parser.add_argument("--hef", action="append", default=[], help="HEF path. Can be passed multiple times.")
    parser.add_argument("--hef-glob", default="", help="Optional glob such as 'build_fixed/*.hef'.")
    parser.add_argument("--save-json", default="", help="Optional path to save full evaluation report.")
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
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    img = Image.fromarray(image_rgb, mode="RGB")
    with BytesIO() as buf:
        img.save(buf, format="JPEG", quality=int(quality))
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_image_blob(image_rgb: np.ndarray, quality: int) -> dict[str, Any]:
    return {
        "encoding": "jpeg_base64",
        "data": _encode_jpeg_base64(image_rgb, quality=quality),
        "shape": list(image_rgb.shape),
    }


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _compare_outputs(ref: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    ref_arr = np.asarray(ref, dtype=np.float32)
    test_arr = np.asarray(test, dtype=np.float32)
    if ref_arr.shape != test_arr.shape:
        return {
            "shape_match": False,
            "ref_shape": list(ref_arr.shape),
            "test_shape": list(test_arr.shape),
        }
    diff = ref_arr - test_arr
    ref_flat = ref_arr.reshape(-1)
    test_flat = test_arr.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(test_flat))
    cosine = float(np.dot(ref_flat, test_flat) / denom) if denom > 0.0 else 1.0
    return {
        "shape_match": True,
        "shape": list(ref_arr.shape),
        "max_abs": float(np.abs(diff).max()),
        "mean_abs": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_signed": float(diff.mean()),
        "cosine_similarity": cosine,
        "allclose": bool(np.allclose(ref_arr, test_arr, rtol=1e-2, atol=1e-2)),
        "allclose_rtol": 1e-2,
        "allclose_atol": 1e-2,
    }


def _select_indices(total: int, num_samples: int, start_index: int, strategy: str, seed: int) -> list[int]:
    if total <= 0:
        return []
    start = max(0, start_index)
    if start >= total:
        raise ValueError(f"start_index={start} is out of range for total={total}")
    available = list(range(start, total))
    if num_samples <= 0 or num_samples >= len(available):
        return available
    if strategy == "first":
        return available[:num_samples]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        picked = rng.choice(np.asarray(available), size=num_samples, replace=False)
        return [int(x) for x in sorted(picked.tolist())]
    positions = np.linspace(0, len(available) - 1, num=num_samples, dtype=np.int64)
    return [available[int(p)] for p in positions.tolist()]


def _build_payload(
    current_rgb: np.ndarray,
    delayed_rgb: np.ndarray,
    projected_tokens: np.ndarray,
    jpeg_quality: int,
) -> dict[str, Any]:
    return {
        "timestamp_ns": time.monotonic_ns(),
        "current_image": _make_image_blob(current_rgb, quality=jpeg_quality),
        "delayed_image": _make_image_blob(delayed_rgb, quality=jpeg_quality),
        "projected_tokens": np.asarray(projected_tokens, dtype=np.float32).tolist(),
    }


def _post_infer(edge_url: str, timeout_s: float, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(edge_url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    body = response.json()
    if not body.get("ok", False):
        raise RuntimeError(f"Remote infer failed: {body}")
    return body


def _summarize_candidate(sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    diff_max_abs: list[float] = []
    diff_mean_abs: list[float] = []
    diff_rmse: list[float] = []
    diff_cosine: list[float] = []
    hef_ms: list[float] = []
    torch_ms: list[float] = []
    total_ms: list[float] = []
    allclose = 0
    shape_mismatch = 0

    for sample in sample_results:
        diff = sample.get("diff_report", {})
        if bool(diff.get("shape_match", False)):
            diff_max_abs.append(float(diff["max_abs"]))
            diff_mean_abs.append(float(diff["mean_abs"]))
            diff_rmse.append(float(diff["rmse"]))
            diff_cosine.append(float(diff["cosine_similarity"]))
            if bool(diff.get("allclose", False)):
                allclose += 1
        else:
            shape_mismatch += 1
        latency = sample.get("latency_ms", {})
        hef_ms.append(float(latency["hef"]))
        torch_ms.append(float(latency["torch"]))
        total_ms.append(float(latency["total"]))

    count = len(sample_results)
    return {
        "samples": count,
        "shape_mismatch_count": shape_mismatch,
        "diff_max_abs_mean": _mean(diff_max_abs),
        "diff_mean_abs_mean": _mean(diff_mean_abs),
        "diff_rmse_mean": _mean(diff_rmse),
        "diff_cosine_similarity_mean": _mean(diff_cosine),
        "allclose_rate": (allclose / count) if count else None,
        "remote_hef_ms_mean": _mean(hef_ms),
        "remote_torch_ms_mean": _mean(torch_ms),
        "remote_total_ms_mean": _mean(total_ms),
    }


def main() -> int:
    args = parse_args()

    if args.check_health:
        healthz = requests.get(_healthz_url(args.edge_url), timeout=args.timeout_s)
        healthz.raise_for_status()

    samples_path = Path(args.samples_json).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    projected_tokens_path = Path(args.projected_tokens).expanduser().resolve()

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    projected_tokens = np.load(projected_tokens_path)
    if len(samples) != int(projected_tokens.shape[0]):
        raise RuntimeError(
            f"samples/token count mismatch: len(samples)={len(samples)} tokens={projected_tokens.shape[0]}"
        )

    hefs: list[Path] = []
    for hef in args.hef:
        hefs.append(Path(hef).expanduser().resolve())
    if args.hef_glob:
        hefs.extend(sorted(Path().glob(args.hef_glob)))
    hefs = list(dict.fromkeys(p.resolve() for p in hefs))
    if not hefs:
        raise RuntimeError("No HEF candidates were provided")

    for hef in hefs:
        if not hef.exists():
            raise FileNotFoundError(f"HEF not found: {hef}")

    indices = _select_indices(
        total=len(samples),
        num_samples=args.num_samples,
        start_index=args.start_index,
        strategy=args.sample_strategy,
        seed=args.seed,
    )
    if not indices:
        raise RuntimeError("No sample indices selected")

    print(f"edge_url: {args.edge_url}")
    print(f"samples_json: {samples_path}")
    print(f"images_dir: {images_dir}")
    print(f"projected_tokens: {projected_tokens_path}")
    print(f"selected_samples: {len(indices)} / {len(samples)}")
    print(f"sample_indices_head: {indices[:10]}")

    local_torch_runner: TorchEdgeRunner | None = None
    if args.compare_reference == "local_torch":
        local_torch_runner = TorchEdgeRunner(
            TorchEdgeRunnerConfig(
                hf_dir=args.hf_dir,
                image_height=args.image_height,
                image_width=args.image_width,
                normalize_imagenet=True,
                image_scale_255=True,
                convert_bgr_to_rgb=False,
                device=args.torch_device,
                dtype=args.torch_dtype,
                preprocess_mode=args.torch_preprocess_mode,
                token_uint8_mode=args.token_quant_mode,
                token_quant_params_path=args.token_quant_params,
            )
        )

    report: dict[str, Any] = {
        "edge_url": args.edge_url,
        "samples_json": str(samples_path),
        "images_dir": str(images_dir),
        "projected_tokens": str(projected_tokens_path),
        "selected_indices": indices,
        "candidates": [],
    }

    for hef in hefs:
        upload = _upload_hef(args.edge_url, hef, args.timeout_s)
        print(f"uploaded_hef: {upload.get('hef_path')}")

        sample_results: list[dict[str, Any]] = []
        for pos, idx in enumerate(indices):
            sample = samples[idx]
            current_path = images_dir / Path(sample["local_image_path"]).name
            if args.delayed_mode == "same":
                delayed_idx = idx
            else:
                delayed_idx = (idx - 1) % len(samples)
            delayed_sample = samples[delayed_idx]
            delayed_path = images_dir / Path(delayed_sample["local_image_path"]).name

            current_rgb = _load_rgb_image(current_path)
            delayed_rgb = _load_rgb_image(delayed_path)
            payload = _build_payload(
                current_rgb=current_rgb,
                delayed_rgb=delayed_rgb,
                projected_tokens=projected_tokens[idx],
                jpeg_quality=args.jpeg_quality,
            )

            if pos < max(args.warmup, 0):
                _post_infer(args.edge_url, args.timeout_s, payload)
                continue

            body = _post_infer(args.edge_url, args.timeout_s, payload)
            diff_report = body.get("diff_report", {})
            if local_torch_runner is not None:
                torch_ref = local_torch_runner.infer(
                    current_image=current_rgb,
                    delayed_image=delayed_rgb,
                    projected_tokens=projected_tokens[idx],
                    goal_pose=None,
                )
                hef_out = np.asarray(body.get("hef_action_chunk"), dtype=np.float32)
                diff_report = _compare_outputs(torch_ref, hef_out)
            sample_results.append(
                {
                    "sample_index": idx,
                    "instruction": sample.get("instruction"),
                    "current_image": str(current_path),
                    "delayed_image": str(delayed_path),
                    "diff_report": diff_report,
                    "latency_ms": body.get("latency_ms", {}),
                    "hef_path": body.get("hef_path"),
                }
            )
            if len(sample_results) % 25 == 0:
                print(f"[{hef.name}] evaluated {len(sample_results)}/{max(0, len(indices) - max(args.warmup, 0))}")

        summary = _summarize_candidate(sample_results)
        candidate_report = {
            "hef": str(hef),
            "uploaded_hef": upload.get("hef_path"),
            "summary": summary,
            "samples": sample_results,
        }
        report["candidates"].append(candidate_report)
        print(json.dumps({"hef": str(hef), **summary}, ensure_ascii=False, indent=2))

    report["ranking"] = sorted(
        [
            {
                "hef": candidate["hef"],
                **candidate["summary"],
            }
            for candidate in report["candidates"]
        ],
        key=lambda item: (
            float("inf") if item["diff_rmse_mean"] is None else float(item["diff_rmse_mean"]),
            float("inf") if item["diff_mean_abs_mean"] is None else float(item["diff_mean_abs_mean"]),
            -(float("-inf") if item["diff_cosine_similarity_mean"] is None else float(item["diff_cosine_similarity_mean"])),
        ),
    )

    print("ranking:")
    print(json.dumps(report["ranking"], ensure_ascii=False, indent=2))

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json: {save_path}")

    if local_torch_runner is not None:
        local_torch_runner.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
