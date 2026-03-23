#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import (
    HailoEdgeRunner,
    HailoEdgeRunnerConfig,
    HybridEdgeRunner,
    HybridEdgeRunnerConfig,
    TorchEdgeRunner,
    TorchEdgeRunnerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local edge runners against a local Torch HF reference using calib_data."
    )
    parser.add_argument("--samples-json", default="calib_data/samples.json")
    parser.add_argument("--images-dir", default="calib_data/images")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "linspace", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delayed-mode", choices=["roll_fullset", "same"], default="same")

    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--torch-preprocess-mode", choices=["hf", "hailo_int8norm"], default="hf")
    parser.add_argument("--token-quant-mode", choices=["dynamic_minmax", "fixed_affine", "none"], default="fixed_affine")
    parser.add_argument("--token-quant-params", default="calib_data/token_quant_fixed_affine_p99_0_uint8.npz")

    parser.add_argument("--hef", action="append", default=[], help="Full HEF path, may be repeated")
    parser.add_argument("--hybrid-hef", action="append", default=[], help="Fused-output HEF path, may be repeated")
    parser.add_argument("--output-fused-name", default="fused_feature")
    parser.add_argument("--fused-dim", type=int, default=1024)
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def _compare_outputs(ref: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    ref_arr = np.asarray(ref, dtype=np.float32)
    test_arr = np.asarray(test, dtype=np.float32)
    if ref_arr.shape != test_arr.shape:
        return {"shape_match": False, "ref_shape": list(ref_arr.shape), "test_shape": list(test_arr.shape)}
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
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    mean_abs = []
    rmse = []
    cosine = []
    latency = []
    for sample in samples:
        diff = sample["diff_report"]
        if diff.get("shape_match", False):
            mean_abs.append(float(diff["mean_abs"]))
            rmse.append(float(diff["rmse"]))
            cosine.append(float(diff["cosine_similarity"]))
        latency.append(float(sample["latency_ms"]))
    return {
        "samples": len(samples),
        "diff_mean_abs_mean": _mean(mean_abs),
        "diff_rmse_mean": _mean(rmse),
        "diff_cosine_similarity_mean": _mean(cosine),
        "runner_ms_mean": _mean(latency),
    }


def _build_full_runner(hef_path: Path, args: argparse.Namespace) -> HailoEdgeRunner:
    return HailoEdgeRunner(
        HailoEdgeRunnerConfig(
            hef_path=str(hef_path),
            input_current_image="current_image",
            input_delayed_image="delayed_image",
            input_projected_tokens="projected_tokens",
            output_action_chunk="depth_to_space1",
            image_height=96,
            image_width=96,
            chunk_size=8,
            pose_dim=4,
            normalize_imagenet=False,
            image_layout="nhwc",
            input_format_type="uint8",
            output_format_type="float32",
            image_scale_255=False,
            convert_bgr_to_rgb=False,
            token_uint8_mode=args.token_quant_mode,
            token_quant_params_path=args.token_quant_params,
        )
    )


def _build_hybrid_runner(hef_path: Path, args: argparse.Namespace) -> HybridEdgeRunner:
    return HybridEdgeRunner(
        HybridEdgeRunnerConfig(
            hef_path=str(hef_path),
            hf_dir=args.hf_dir,
            output_fused_feature=args.output_fused_name,
            fused_dim=args.fused_dim,
            normalize_imagenet=False,
            image_layout="nhwc",
            input_format_type="uint8",
            output_format_type="float32",
            image_scale_255=False,
            convert_bgr_to_rgb=False,
            token_uint8_mode=args.token_quant_mode,
            token_quant_params_path=args.token_quant_params,
            device=args.torch_device,
            dtype=args.torch_dtype,
        )
    )


def main() -> int:
    args = parse_args()
    samples = json.loads(Path(args.samples_json).expanduser().resolve().read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir).expanduser().resolve()
    projected_tokens = np.load(Path(args.projected_tokens).expanduser().resolve())
    if len(samples) != int(projected_tokens.shape[0]):
        raise RuntimeError(
            f"samples/token count mismatch: len(samples)={len(samples)} tokens={projected_tokens.shape[0]}"
        )

    indices = _select_indices(len(samples), args.num_samples, args.start_index, args.sample_strategy, args.seed)

    reference = TorchEdgeRunner(
        TorchEdgeRunnerConfig(
            hf_dir=args.hf_dir,
            image_height=96,
            image_width=96,
            device=args.torch_device,
            dtype=args.torch_dtype,
            preprocess_mode=args.torch_preprocess_mode,
            token_uint8_mode=args.token_quant_mode,
            token_quant_params_path=args.token_quant_params,
        )
    )

    runners: list[tuple[str, Any]] = []
    for hef in args.hef:
        path = Path(hef).expanduser().resolve()
        runners.append((path.name, _build_full_runner(path, args)))
    for hef in args.hybrid_hef:
        path = Path(hef).expanduser().resolve()
        runners.append((path.name + ":hybrid", _build_hybrid_runner(path, args)))
    if not runners:
        raise RuntimeError("No --hef or --hybrid-hef specified")

    report: dict[str, Any] = {
        "config": {
            "samples_json": str(Path(args.samples_json).expanduser().resolve()),
            "images_dir": str(images_dir),
            "projected_tokens": str(Path(args.projected_tokens).expanduser().resolve()),
            "num_samples": len(indices),
            "delayed_mode": args.delayed_mode,
            "torch_preprocess_mode": args.torch_preprocess_mode,
            "token_quant_mode": args.token_quant_mode,
            "token_quant_params": args.token_quant_params,
        },
        "candidates": [],
    }

    try:
        for name, runner in runners:
            sample_reports = []
            for sample_idx in indices:
                current_path = images_dir / samples[sample_idx]["image"]
                current_rgb = _load_rgb_image(current_path)
                if args.delayed_mode == "same":
                    delayed_rgb = current_rgb
                else:
                    delayed_idx = (sample_idx - 1) % len(samples)
                    delayed_rgb = _load_rgb_image(images_dir / samples[delayed_idx]["image"])
                tokens = np.asarray(projected_tokens[sample_idx], dtype=np.float32)

                ref = reference.infer(current_rgb, delayed_rgb, tokens)
                t0 = time.perf_counter()
                out = runner.infer(current_rgb, delayed_rgb, tokens)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                sample_reports.append(
                    {
                        "sample_index": int(sample_idx),
                        "diff_report": _compare_outputs(ref, out),
                        "latency_ms": latency_ms,
                    }
                )
            report["candidates"].append(
                {
                    "candidate": name,
                    "summary": _summarize(sample_reports),
                    "samples": sample_reports,
                }
            )
    finally:
        reference.close()
        for _, runner in runners:
            runner.close()

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
