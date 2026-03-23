#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


DEFAULT_POINTS: list[tuple[str, str, str]] = [
    ("cat_pool", "/_avg_pooling/GlobalAveragePool_output_0", "/_avg_pooling/GlobalAveragePool_output_0"),
    ("obs_pool", "/_avg_pooling_1/GlobalAveragePool_output_0", "/_avg_pooling_1/GlobalAveragePool_output_0"),
    ("cat_compress", "/compress_cat_enc/Gemm_output_0", "/compress_cat_enc/Gemm_output_0"),
    ("obs_compress", "/compress_obs_enc/Gemm_output_0", "/compress_obs_enc/Gemm_output_0"),
    ("decoder_pos_add", "/decoder/positional_encoding/Add_output_0", "/decoder/positional_encoding/Add_output_0"),
    ("decoder_l0", "/decoder/sa_decoder/layers.0/Add_2_output_0", "/decoder/sa_decoder/layers.0/Add_2_output_0"),
    ("decoder_l1", "/decoder/sa_decoder/layers.1/Add_2_output_0", "/decoder/sa_decoder/layers.1/Add_2_output_0"),
    ("decoder_l2", "/decoder/sa_decoder/layers.2/Add_2_output_0", "/decoder/sa_decoder/layers.2/Add_2_output_0"),
    ("decoder_l3", "/decoder/sa_decoder/layers.3/Add_2_output_0", "/decoder/sa_decoder/layers.3/Add_2_output_0"),
    ("mlp_relu0", "/action_predictor/action_predictor.1/Relu_output_0", "/action_predictor/action_predictor.1/Relu_output_0"),
    ("mlp_relu1", "/action_predictor/action_predictor.3/Relu_output_0", "/action_predictor/action_predictor.3/Relu_output_0"),
    ("mlp_relu2", "/action_predictor/action_predictor.5/Relu_output_0", "/action_predictor/action_predictor.5/Relu_output_0"),
    ("action_out", "/action_predictor/action_predictor.6/Gemm_output_0", "/action_predictor/action_predictor.6/Gemm_output_0"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layer-wise fp32 vs static-int8 ONNX differences for edge adapter.")
    parser.add_argument("--fp32-onnx", default="build_fixed/edge_adapter_fp32.onnx")
    parser.add_argument("--int8-onnx", default="build_fixed/edge_adapter_static_int8.onnx")
    parser.add_argument("--samples-json", default="calib_data/samples.json")
    parser.add_argument("--images-dir", default="calib_data/images")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "linspace", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delayed-mode", choices=["same", "roll_fullset"], default="same")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--save-json", default="build_fixed/edge_onnx_layerwise_diff.json")
    return parser.parse_args()


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
    pil = Image.fromarray(image, mode="RGB")
    return np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _preprocess_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = _resize_image(image, width=width, height=height).astype(np.float32)
    resized = resized / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std
    return np.transpose(resized, (2, 0, 1))


def _make_sample_inputs(
    idx: int,
    samples: list[dict[str, Any]],
    projected_tokens: np.ndarray,
    images_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    sample = samples[idx]
    current_path = images_dir / Path(sample["local_image_path"]).name
    delayed_idx = idx if args.delayed_mode == "same" else (idx - 1) % len(samples)
    delayed_path = images_dir / Path(samples[delayed_idx]["local_image_path"]).name
    current_rgb = _load_rgb_image(current_path)
    delayed_rgb = _load_rgb_image(delayed_path)
    current = _preprocess_image(current_rgb, args.image_width, args.image_height)[None, ...]
    delayed = _preprocess_image(delayed_rgb, args.image_width, args.image_height)[None, ...]
    tokens = np.asarray(projected_tokens[idx], dtype=np.float32)[None, ...]
    return {
        "current_image": current.astype(np.float32, copy=False),
        "delayed_image": delayed.astype(np.float32, copy=False),
        "projected_tokens": tokens,
    }


def _available_outputs(model: onnx.ModelProto) -> set[str]:
    return {out for node in model.graph.node for out in node.output}


def _find_value_info(model: onnx.ModelProto, tensor_name: str) -> onnx.ValueInfoProto | None:
    for value in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        if value.name == tensor_name:
            return value
    return None


def _augment_model(model_path: Path, output_names: list[str]) -> Path:
    model = onnx.load(str(model_path))
    available = _available_outputs(model)
    missing = [name for name in output_names if name not in available]
    if missing:
        raise KeyError(f"Missing outputs in {model_path}: {missing}")
    existing = {out.name for out in model.graph.output}
    for name in output_names:
        if name in existing:
            continue
        value_info = _find_value_info(model, name)
        if value_info is None:
            value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        model.graph.output.append(value_info)
    tmp = tempfile.NamedTemporaryFile(prefix=model_path.stem + "_aug_", suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    diff = a32 - b32
    denom = float(np.linalg.norm(a32.reshape(-1)) * np.linalg.norm(b32.reshape(-1)))
    cosine = float(np.dot(a32.reshape(-1), b32.reshape(-1)) / denom) if denom > 0.0 else 1.0
    return {
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> int:
    args = parse_args()
    fp32_path = Path(args.fp32_onnx).expanduser().resolve()
    int8_path = Path(args.int8_onnx).expanduser().resolve()
    samples = json.loads(Path(args.samples_json).expanduser().resolve().read_text(encoding="utf-8"))
    projected_tokens = np.asarray(np.load(Path(args.projected_tokens).expanduser().resolve()), dtype=np.float32)
    images_dir = Path(args.images_dir).expanduser().resolve()

    selected = _select_indices(len(samples), args.num_samples, args.start_index, args.sample_strategy, args.seed)

    fp_names = [fp for _, fp, _ in DEFAULT_POINTS]
    int8_names = [iq for _, _, iq in DEFAULT_POINTS]
    fp_aug = _augment_model(fp32_path, fp_names)
    int8_aug = _augment_model(int8_path, int8_names)

    fp_sess = ort.InferenceSession(str(fp_aug), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_aug), providers=["CPUExecutionProvider"])
    fp_out_names = [o.name for o in fp_sess.get_outputs()]
    int8_out_names = [o.name for o in int8_sess.get_outputs()]

    fp_index = {name: idx for idx, name in enumerate(fp_out_names)}
    int8_index = {name: idx for idx, name in enumerate(int8_out_names)}

    point_metrics: dict[str, dict[str, list[float]]] = {
        label: {"mean_abs": [], "rmse": [], "max_abs": [], "cosine": []} for label, _, _ in DEFAULT_POINTS
    }

    for idx in selected:
        feeds = _make_sample_inputs(idx, samples, projected_tokens, images_dir, args)
        fp_vals = fp_sess.run(None, feeds)
        int8_vals = int8_sess.run(None, feeds)
        for label, fp_name, int8_name in DEFAULT_POINTS:
            fp_val = fp_vals[fp_index[fp_name]]
            int8_val = int8_vals[int8_index[int8_name]]
            stats = _compare(fp_val, int8_val)
            for k, v in stats.items():
                point_metrics[label][k].append(float(v))

    ordered = []
    for label, _, _ in DEFAULT_POINTS:
        summary = {k: _mean(v) for k, v in point_metrics[label].items()}
        ordered.append({"point": label, **summary})

    prev_rmse = 0.0
    prev_mean_abs = 0.0
    for item in ordered:
        item["delta_rmse_vs_prev"] = float(item["rmse"] - prev_rmse)
        item["delta_mean_abs_vs_prev"] = float(item["mean_abs"] - prev_mean_abs)
        prev_rmse = float(item["rmse"])
        prev_mean_abs = float(item["mean_abs"])

    rmse_surge = max(ordered, key=lambda x: x["delta_rmse_vs_prev"])
    mean_abs_surge = max(ordered, key=lambda x: x["delta_mean_abs_vs_prev"])

    report = {
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "samples": len(selected),
        "sample_indices_head": selected[:10],
        "ordered_points": ordered,
        "largest_rmse_surge": rmse_surge,
        "largest_mean_abs_surge": mean_abs_surge,
        "notes": [
            "This is fp32 ONNX vs static-int8 ONNX layer-wise analysis, not direct HEF internal activation comparison.",
            "It identifies blocks most sensitive to 8bit quantization under the same HF preprocessing.",
        ],
    }

    out = Path(args.save_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
