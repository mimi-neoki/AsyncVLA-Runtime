#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi.token_quant import build_token_quant_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed affine token quantization params from calibration tokens.")
    parser.add_argument("--projected-tokens", default="calib_data/calib_projected_tokens_8x1024_float32.npy")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--quant-dtype", choices=("uint8", "int8"), default="uint8")
    parser.add_argument(
        "--output",
        default="calib_data/token_quant_fixed_affine_p99_9_uint8.npz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projected_tokens = np.load(Path(args.projected_tokens).expanduser().resolve())
    params = build_token_quant_params(
        projected_tokens,
        percentile=float(args.percentile),
        quant_dtype=args.quant_dtype,
    )
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        scales=np.asarray(params["scales"], dtype=np.float32),
        zero_point=np.int32(params["zero_point"]),
        percentile=np.float32(params["percentile"]),
        quant_dtype=np.array(args.quant_dtype),
    )
    print(f"saved: {out_path}")
    print(
        {
            "percentile": float(args.percentile),
            "quant_dtype": args.quant_dtype,
            "scales_min": float(np.min(params["scales"])),
            "scales_max": float(np.max(params["scales"])),
            "scales_mean": float(np.mean(params["scales"])),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
