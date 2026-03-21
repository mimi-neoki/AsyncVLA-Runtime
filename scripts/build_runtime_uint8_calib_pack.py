#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a runtime-aligned quantized calibration pack for the edge adapter."
    )
    parser.add_argument("--calib-dir", default="calib_data")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--quant-dtype", choices=("uint8", "int8"), default="uint8")
    parser.add_argument(
        "--delayed-mode",
        choices=("roll", "same"),
        default="roll",
        help="How to synthesize the delayed image input from the calibration image set.",
    )
    parser.add_argument(
        "--output-name",
        default="edge_adapter_calib_inputs_runtime_{dtype}_{mode}_n{n}.npz",
    )
    return parser.parse_args()


def _quantize_tokens_dynamic_minmax(tokens: np.ndarray, quant_dtype: str) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected projected tokens with shape [N, 8, 1024], got {arr.shape}")
    t_min = arr.min(axis=(1, 2), keepdims=True)
    t_max = arr.max(axis=(1, 2), keepdims=True)
    denom = np.maximum(t_max - t_min, 1e-6)
    if quant_dtype == "int8":
        scaled = (arr - t_min) / denom * 255.0 - 128.0
        return np.clip(np.round(scaled), -128, 127).astype(np.int8)
    return np.clip(np.round((arr - t_min) / denom * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    calib_dir = Path(args.calib_dir).expanduser().resolve()

    images_u8_path = calib_dir / "calib_images_96x96_uint8.npy"
    projected_tokens_path = calib_dir / "calib_projected_tokens_8x1024_float32.npy"

    images_u8 = np.load(images_u8_path)
    projected_tokens = np.load(projected_tokens_path)

    n = min(int(args.num_samples), int(images_u8.shape[0]), int(projected_tokens.shape[0]))
    if n <= 0:
        raise RuntimeError("No samples available")

    current_u8 = np.asarray(images_u8[:n], dtype=np.uint8)
    if args.delayed_mode == "same":
        delayed_u8 = current_u8.copy()
    else:
        delayed_u8 = np.roll(current_u8, shift=1, axis=0)

    if args.quant_dtype == "int8":
        current_q = np.clip(current_u8.astype(np.int16) - 128, -128, 127).astype(np.int8)
        delayed_q = np.clip(delayed_u8.astype(np.int16) - 128, -128, 127).astype(np.int8)
    else:
        current_q = current_u8
        delayed_q = delayed_u8
    projected_q = _quantize_tokens_dynamic_minmax(projected_tokens[:n], args.quant_dtype)[:, None, :, :]

    output_name = args.output_name.format(n=n, mode=args.delayed_mode, dtype=args.quant_dtype)
    out_path = calib_dir / output_name
    np.savez(
        out_path,
        **{
            "edge_adapter_static/input_layer1": current_q,
            "edge_adapter_static/input_layer2": delayed_q,
            "edge_adapter_static/input_layer3": projected_q,
        },
    )

    print(f"saved: {out_path}")
    print(
        "shapes:",
        {
            "input_layer1": list(current_q.shape),
            "input_layer2": list(delayed_q.shape),
            "input_layer3": list(projected_q.shape),
        },
    )
    print(
        "ranges:",
        {
            "input_layer1": [int(current_q.min()), int(current_q.max())],
            "input_layer2": [int(delayed_q.min()), int(delayed_q.max())],
            "input_layer3": [int(projected_q.min()), int(projected_q.max())],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
