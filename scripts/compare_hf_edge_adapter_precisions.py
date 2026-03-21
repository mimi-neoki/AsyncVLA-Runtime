#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi.edge_adapter_model import load_edge_adapter_from_hf_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF edge adapter outputs between fp16 PyTorch and static-int8 ONNXRuntime."
    )
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--checkpoint-name", default="shead--750000_checkpoint.pt")
    parser.add_argument("--mha-num-attention-heads", type=int, default=4)
    parser.add_argument("--onnx-path", default="build_fixed/edge_adapter_fp32.onnx")
    parser.add_argument("--int8-onnx-path", default="build_fixed/edge_adapter_static_int8.onnx")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-quantize", action="store_true")

    parser.add_argument("--samples-json", default="calib_data/samples.json")
    parser.add_argument("--images-dir", default="calib_data/images")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "linspace", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delayed-mode", choices=["roll_fullset", "same"], default="roll_fullset")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--normalize-imagenet", action="store_true", default=True)
    parser.add_argument("--no-normalize-imagenet", dest="normalize_imagenet", action="store_false")
    parser.add_argument("--image-scale-255", action="store_true", default=True)
    parser.add_argument("--no-image-scale-255", dest="image_scale_255", action="store_false")

    parser.add_argument("--quant-format", choices=["qdq", "qoperator"], default="qdq")
    parser.add_argument("--activation-type", choices=["qint8", "quint8"], default="quint8")
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--calibrate-method", choices=["minmax", "entropy", "percentile"], default="minmax")

    parser.add_argument("--save-json", default="build_fixed/compare_hf_edge_adapter_precisions.json")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_quant_format(name: str) -> QuantFormat:
    if name == "qdq":
        return QuantFormat.QDQ
    if name == "qoperator":
        return QuantFormat.QOperator
    raise ValueError(f"Unsupported quant format: {name}")


def _resolve_quant_type(name: str) -> QuantType:
    if name == "qint8":
        return QuantType.QInt8
    if name == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported quant type: {name}")


def _resolve_calibrate_method(name: str) -> CalibrationMethod:
    if name == "minmax":
        return CalibrationMethod.MinMax
    if name == "entropy":
        return CalibrationMethod.Entropy
    if name == "percentile":
        return CalibrationMethod.Percentile
    raise ValueError(f"Unsupported calibrate method: {name}")


def _select_indices(total: int, num_samples: int, start_index: int, strategy: str, seed: int) -> list[int]:
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


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, (width, height))
    pil = Image.fromarray(image, mode="RGB")
    return np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _preprocess_image(image: np.ndarray, width: int, height: int, image_scale_255: bool, normalize_imagenet: bool) -> np.ndarray:
    resized = _resize_image(image, width=width, height=height).astype(np.float32)
    if image_scale_255 and resized.max() > 1.0:
        resized = resized / 255.0
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        resized = (resized - mean) / std
    return np.transpose(resized, (2, 0, 1))


def _load_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    samples = json.loads(Path(args.samples_json).expanduser().resolve().read_text(encoding="utf-8"))
    projected_tokens = np.load(Path(args.projected_tokens).expanduser().resolve())
    if len(samples) != int(projected_tokens.shape[0]):
        raise RuntimeError(
            f"samples/token count mismatch: len(samples)={len(samples)} tokens={projected_tokens.shape[0]}"
        )
    return samples, np.asarray(projected_tokens, dtype=np.float32)


def _make_sample_inputs(
    idx: int,
    samples: list[dict[str, Any]],
    projected_tokens: np.ndarray,
    images_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    sample = samples[idx]
    current_path = images_dir / Path(sample["local_image_path"]).name
    if args.delayed_mode == "same":
        delayed_idx = idx
    else:
        delayed_idx = (idx - 1) % len(samples)
    delayed_path = images_dir / Path(samples[delayed_idx]["local_image_path"]).name

    current_rgb = _load_rgb_image(current_path)
    delayed_rgb = _load_rgb_image(delayed_path)

    current = _preprocess_image(
        current_rgb,
        width=args.image_width,
        height=args.image_height,
        image_scale_255=args.image_scale_255,
        normalize_imagenet=args.normalize_imagenet,
    )[None, ...]
    delayed = _preprocess_image(
        delayed_rgb,
        width=args.image_width,
        height=args.image_height,
        image_scale_255=args.image_scale_255,
        normalize_imagenet=args.normalize_imagenet,
    )[None, ...]
    tokens = np.asarray(projected_tokens[idx], dtype=np.float32)[None, ...]
    return {
        "current_image": current.astype(np.float32, copy=False),
        "delayed_image": delayed.astype(np.float32, copy=False),
        "projected_tokens": tokens,
    }


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, samples: list[dict[str, np.ndarray]]) -> None:
        self._samples = list(samples)
        self._iter: Any = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self._samples)
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = None


def _ensure_exported_onnx(args: argparse.Namespace, onnx_path: Path) -> None:
    if onnx_path.exists() and not args.force_export:
        return
    model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
        hf_dir=args.hf_dir,
        checkpoint_name=args.checkpoint_name,
        mha_num_attention_heads=args.mha_num_attention_heads,
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")
    model = model.to(dtype=torch.float32, device="cpu")
    model.eval()

    dummy_current = torch.randn(1, 3, args.image_height, args.image_width, dtype=torch.float32)
    dummy_delayed = torch.randn(1, 3, args.image_height, args.image_width, dtype=torch.float32)
    dummy_tokens = torch.randn(1, arch.action_chunk_size, arch.obs_encoding_size, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_current, dummy_delayed, dummy_tokens),
        str(onnx_path),
        input_names=["current_image", "delayed_image", "projected_tokens"],
        output_names=["action_chunk"],
        opset_version=17,
    )


def _ensure_quantized_onnx(
    args: argparse.Namespace,
    onnx_path: Path,
    int8_onnx_path: Path,
    calib_inputs: list[dict[str, np.ndarray]],
) -> None:
    if int8_onnx_path.exists() and not args.force_quantize:
        return
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=_CalibrationReader(calib_inputs),
        quant_format=_resolve_quant_format(args.quant_format),
        per_channel=bool(args.per_channel),
        activation_type=_resolve_quant_type(args.activation_type),
        weight_type=_resolve_quant_type(args.weight_type),
        calibrate_method=_resolve_calibrate_method(args.calibrate_method),
        extra_options={"ForceQuantizeNoInputCheck": True},
    )


def _compare_outputs(ref: np.ndarray, test: np.ndarray) -> dict[str, float | bool | list[int]]:
    ref_arr = np.asarray(ref, dtype=np.float32)
    test_arr = np.asarray(test, dtype=np.float32)
    diff = ref_arr - test_arr
    ref_flat = ref_arr.reshape(-1)
    test_flat = test_arr.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(test_flat))
    cosine = float(np.dot(ref_flat, test_flat) / denom) if denom > 0.0 else 1.0
    return {
        "shape": list(ref_arr.shape),
        "max_abs": float(np.abs(diff).max()),
        "mean_abs": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_signed": float(diff.mean()),
        "cosine_similarity": cosine,
        "allclose": bool(np.allclose(ref_arr, test_arr, rtol=1e-2, atol=1e-2)),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    int8_onnx_path = Path(args.int8_onnx_path).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()

    samples, projected_tokens = _load_dataset(args)
    calib_indices = _select_indices(len(samples), args.calib_samples, 0, args.sample_strategy, args.seed)
    eval_indices = _select_indices(len(samples), args.num_samples, args.start_index, args.sample_strategy, args.seed)

    calib_inputs = [_make_sample_inputs(i, samples, projected_tokens, images_dir, args) for i in calib_indices]

    _ensure_exported_onnx(args, onnx_path)
    _ensure_quantized_onnx(args, onnx_path, int8_onnx_path, calib_inputs)

    model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
        hf_dir=args.hf_dir,
        checkpoint_name=args.checkpoint_name,
        mha_num_attention_heads=args.mha_num_attention_heads,
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.fp_dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU execution only supports --fp-dtype float32")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    ort_session = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])

    sample_reports: list[dict[str, Any]] = []
    fp_ms_values: list[float] = []
    int8_ms_values: list[float] = []
    rmse_values: list[float] = []
    mean_abs_values: list[float] = []
    cosine_values: list[float] = []

    with torch.inference_mode():
        for idx in eval_indices:
            tensors = _make_sample_inputs(idx, samples, projected_tokens, images_dir, args)
            current_t = torch.from_numpy(tensors["current_image"]).to(device=device, dtype=dtype)
            delayed_t = torch.from_numpy(tensors["delayed_image"]).to(device=device, dtype=dtype)
            tokens_t = torch.from_numpy(tensors["projected_tokens"]).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            fp_out = model(current_t, delayed_t, tokens_t).detach().cpu().to(torch.float32).numpy()
            fp_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            int8_out = ort_session.run(None, tensors)[0]
            int8_ms = (time.perf_counter() - t1) * 1000.0

            diff = _compare_outputs(fp_out, int8_out)
            fp_ms_values.append(fp_ms)
            int8_ms_values.append(int8_ms)
            rmse_values.append(float(diff["rmse"]))
            mean_abs_values.append(float(diff["mean_abs"]))
            cosine_values.append(float(diff["cosine_similarity"]))
            sample_reports.append(
                {
                    "sample_index": idx,
                    "instruction": samples[idx].get("instruction"),
                    "diff_report": diff,
                    "latency_ms": {
                        "fp16_torch": fp_ms,
                        "int8_onnxruntime": int8_ms,
                    },
                }
            )

    report = {
        "hf_dir": str(Path(args.hf_dir).expanduser().resolve()),
        "checkpoint_name": args.checkpoint_name,
        "onnx_path": str(onnx_path),
        "int8_onnx_path": str(int8_onnx_path),
        "device": str(device),
        "fp_dtype": args.fp_dtype,
        "quant_format": args.quant_format,
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "per_channel": bool(args.per_channel),
        "calibrate_method": args.calibrate_method,
        "architecture": {
            "obs_encoding_size": int(arch.obs_encoding_size),
            "seq_len": int(arch.seq_len),
            "mha_num_attention_layers": int(arch.mha_num_attention_layers),
            "mha_ff_dim_factor": int(arch.mha_ff_dim_factor),
            "action_chunk_size": int(arch.action_chunk_size),
            "action_dim": int(arch.action_dim),
        },
        "calibration_samples": len(calib_indices),
        "evaluation_samples": len(eval_indices),
        "summary": {
            "fp16_torch_ms_mean": _mean(fp_ms_values),
            "int8_onnxruntime_ms_mean": _mean(int8_ms_values),
            "diff_mean_abs_mean": _mean(mean_abs_values),
            "diff_rmse_mean": _mean(rmse_values),
            "diff_cosine_similarity_mean": _mean(cosine_values),
            "allclose_rate": float(sum(1 for r in sample_reports if r["diff_report"]["allclose"]) / len(sample_reports)),
        },
        "samples": sample_reports,
    }

    save_path = Path(args.save_json).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved_json: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
