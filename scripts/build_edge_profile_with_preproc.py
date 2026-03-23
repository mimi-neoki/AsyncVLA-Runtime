#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi.token_quant import load_token_quant_params


def _fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.8g}" for v in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hailo model-script profile with embedded preprocessing.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-quant-params", required=True)
    parser.add_argument("--model-prefix", default="edge_adapter_static")
    parser.add_argument("--current-input-name", default=None)
    parser.add_argument("--delayed-input-name", default=None)
    parser.add_argument("--tokens-input-name", default=None)
    parser.add_argument("--image-preproc", choices=["hf", "int8norm"], default="hf")
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--compression-level", type=int, default=2)
    parser.add_argument("--compiler-optimization-level", default="1")
    parser.add_argument("--calibset-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_params = load_token_quant_params(args.token_quant_params)
    scales = np.asarray(token_params["scales"], dtype=np.float32)
    zero_point = float(token_params["zero_point"])

    if args.image_preproc == "hf":
        image_mean = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]
        image_std = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    else:
        image_mean = [128.0, 128.0, 128.0]
        image_std = [1.0, 1.0, 1.0]

    token_mean = [zero_point] * int(scales.shape[0])
    token_std = [float(1.0 / s) for s in scales]

    model_prefix = str(args.model_prefix).strip().rstrip("/")
    current_input_name = str(args.current_input_name).strip() if args.current_input_name else ""
    delayed_input_name = str(args.delayed_input_name).strip() if args.delayed_input_name else ""
    tokens_input_name = str(args.tokens_input_name).strip() if args.tokens_input_name else ""

    if not current_input_name:
        if not model_prefix:
            raise ValueError("Either --model-prefix or explicit input names must be provided")
        current_input_name = f"{model_prefix}/input_layer1"
    if not delayed_input_name:
        if not model_prefix:
            raise ValueError("Either --model-prefix or explicit input names must be provided")
        delayed_input_name = f"{model_prefix}/input_layer2"
    if not tokens_input_name:
        if not model_prefix:
            raise ValueError("Either --model-prefix or explicit input names must be provided")
        tokens_input_name = f"{model_prefix}/input_layer3"

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"cur_input_center = normalization({_fmt_list(image_mean)}, {_fmt_list(image_std)}, {current_input_name})",
            f"past_input_center = normalization({_fmt_list(image_mean)}, {_fmt_list(image_std)}, {delayed_input_name})",
            f"token_input_center = normalization({_fmt_list(token_mean)}, {_fmt_list(token_std)}, {tokens_input_name})",
            f"model_optimization_flavor(optimization_level={args.optimization_level}, compression_level={args.compression_level})",
            f"performance_param(compiler_optimization_level={args.compiler_optimization_level})",
            f"model_optimization_config(calibration, calibset_size={args.calibset_size})",
            "",
        ]
    )
    out.write_text(text)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
