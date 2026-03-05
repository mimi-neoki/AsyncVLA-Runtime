#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi.edge_adapter_model import load_edge_adapter_from_hf_snapshot


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load AsyncVLA edge adapter from HF snapshot, export ONNX, and optionally run Hailo checks"
    )
    parser.add_argument(
        "--hf-dir",
        default="~/gitrepo/AsyncVLA_release",
        help="Path to HF snapshot directory",
    )
    parser.add_argument(
        "--shead-checkpoint",
        default="shead--750000_checkpoint.pt",
        help="Edge adapter checkpoint filename",
    )
    parser.add_argument(
        "--mha-num-attention-heads",
        type=int,
        default=4,
        help="Head count used to instantiate the Transformer decoder",
    )
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument(
        "--onnx-output",
        default="./build/edge_adapter.onnx",
        help="Output ONNX path for export check",
    )
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-hailo", action="store_true")
    parser.add_argument(
        "--h10-hef",
        default="/usr/share/hailo-models/resnet_v1_50_h10.hef",
        help="HEF path for Hailo-10H runtime smoke check",
    )
    parser.add_argument("--benchmark-seconds", type=int, default=3)
    return parser.parse_args()


def check_onnx_equivalence(
    onnx_path: Path,
    model: torch.nn.Module,
    current_image: torch.Tensor,
    delayed_image: torch.Tensor,
    projected_tokens: torch.Tensor,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> tuple[float, float]:
    with torch.inference_mode():
        torch_out = model(current_image, delayed_image, projected_tokens).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {
            "current_image": current_image.cpu().numpy(),
            "delayed_image": delayed_image.cpu().numpy(),
            "projected_tokens": projected_tokens.cpu().numpy(),
        },
    )[0]
    diff = np.abs(torch_out - onnx_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    if not np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"ONNX mismatch: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, rtol={rtol}, atol={atol}"
        )
    return max_abs, mean_abs


def main() -> int:
    args = parse_args()
    hf_dir = Path(args.hf_dir).expanduser().resolve()
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF snapshot not found: {hf_dir}")

    model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
        hf_dir=hf_dir,
        checkpoint_name=args.shead_checkpoint,
        mha_num_attention_heads=args.mha_num_attention_heads,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(f"Unexpected key mismatch: missing={missing}, unexpected={unexpected}")

    torch.manual_seed(7)
    current_image = torch.randn(1, 3, args.image_height, args.image_width, dtype=torch.float32)
    delayed_image = torch.randn(1, 3, args.image_height, args.image_width, dtype=torch.float32)
    projected_tokens = torch.randn(1, arch.action_chunk_size, arch.obs_encoding_size, dtype=torch.float32)

    with torch.inference_mode():
        action_chunk = model(current_image, delayed_image, projected_tokens)

    print("[Edge adapter load] OK")
    print(f"hf_dir: {hf_dir}")
    print(
        "arch:",
        {
            "obs_encoding_size": arch.obs_encoding_size,
            "seq_len": arch.seq_len,
            "mha_layers": arch.mha_num_attention_layers,
            "mha_ff_factor": arch.mha_ff_dim_factor,
            "action_chunk_size": arch.action_chunk_size,
            "action_dim": arch.action_dim,
        },
    )
    print(f"shead missing/unexpected: {len(missing)}/{len(unexpected)}")
    print(f"action_chunk shape: {tuple(action_chunk.shape)}")
    print(
        "action_chunk stats:",
        {
            "min": float(action_chunk.min()),
            "max": float(action_chunk.max()),
            "mean": float(action_chunk.mean()),
        },
    )

    if not args.skip_onnx:
        onnx_path = Path(args.onnx_output).expanduser().resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            (current_image, delayed_image, projected_tokens),
            str(onnx_path),
            input_names=["current_image", "delayed_image", "projected_tokens"],
            output_names=["action_chunk"],
            dynamic_axes={
                "current_image": {0: "batch"},
                "delayed_image": {0: "batch"},
                "projected_tokens": {0: "batch"},
                "action_chunk": {0: "batch"},
            },
            opset_version=17,
        )
        max_abs, mean_abs = check_onnx_equivalence(
            onnx_path=onnx_path,
            model=model,
            current_image=current_image,
            delayed_image=delayed_image,
            projected_tokens=projected_tokens,
        )
        print(f"[ONNX export] OK: {onnx_path}")
        print(f"onnx parity: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")

    if args.skip_hailo:
        return 0

    h10_hef = Path(args.h10_hef).expanduser().resolve()
    if not h10_hef.exists():
        raise FileNotFoundError(f"H10 HEF not found: {h10_hef}")

    print("\n[Hailo-10H runtime check]")
    scan = run_cmd(["hailortcli", "scan"])
    identify = run_cmd(["hailortcli", "fw-control", "identify"])
    parse = run_cmd(["hailortcli", "parse-hef", str(h10_hef)])
    bench = run_cmd(
        [
            "hailortcli",
            "benchmark",
            str(h10_hef),
            "--time-to-run",
            str(args.benchmark_seconds),
            "--batch-size",
            "1",
        ]
    )

    print(scan.stdout.strip())
    print(identify.stdout.strip())
    print(parse.stdout.strip().splitlines()[0])
    bench_lines = [line for line in bench.stdout.strip().splitlines() if line.strip()]
    print("benchmark tail:")
    for line in bench_lines[-12:]:
        print(line)

    print(
        "\nNOTE: Hailo execution above validates Hailo-10H runtime path. "
        "Edge adapter deployment requires compiling this exported ONNX to HEF on x86 with Hailo Dataflow Compiler."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
