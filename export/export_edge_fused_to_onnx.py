#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi.edge_adapter_model import EdgeAdapterFusedBackbone, load_edge_adapter_from_hf_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AsyncVLA edge adapter up to fused decoder latent to ONNX")
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--shead-checkpoint", default="shead--750000_checkpoint.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--mha-num-attention-heads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-onnxruntime", action="store_true")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser.parse_args()


def _dummy_inputs(
    batch_size: int,
    image_height: int,
    image_width: int,
    chunk_size: int,
    projected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_image = torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32)
    delayed_image = torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32)
    projected_tokens = torch.randn(batch_size, chunk_size, projected_dim, dtype=torch.float32)
    return current_image, delayed_image, projected_tokens


def _verify_with_onnxruntime(
    onnx_path: Path,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rtol: float,
    atol: float,
) -> None:
    import onnxruntime as ort

    current_image, delayed_image, projected_tokens = inputs
    with torch.inference_mode():
        torch_out = model(current_image, delayed_image, projected_tokens).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(
        None,
        {
            "current_image": current_image.cpu().numpy(),
            "delayed_image": delayed_image.cpu().numpy(),
            "projected_tokens": projected_tokens.cpu().numpy(),
        },
    )[0]

    if not np.allclose(torch_out, ort_out, rtol=rtol, atol=atol):
        diff = np.abs(torch_out - ort_out)
        raise RuntimeError(
            "ONNXRuntime output mismatch: "
            f"max_abs={float(diff.max()):.6f}, mean_abs={float(diff.mean()):.6f}, "
            f"rtol={rtol}, atol={atol}"
        )


def main() -> None:
    args = parse_args()
    model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
        hf_dir=args.hf_dir,
        checkpoint_name=args.shead_checkpoint,
        mha_num_attention_heads=args.mha_num_attention_heads,
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch detected even with strict=True: missing={missing}, unexpected={unexpected}"
        )
    fused_model = EdgeAdapterFusedBackbone(model)
    fused_model.eval()

    inputs = _dummy_inputs(
        batch_size=args.batch_size,
        image_height=args.image_height,
        image_width=args.image_width,
        chunk_size=arch.action_chunk_size,
        projected_dim=arch.obs_encoding_size,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        fused_model,
        inputs,
        str(output_path),
        input_names=["current_image", "delayed_image", "projected_tokens"],
        output_names=["fused_feature"],
        opset_version=args.opset,
    )

    if args.verify_onnxruntime:
        _verify_with_onnxruntime(
            onnx_path=output_path,
            model=fused_model,
            inputs=inputs,
            rtol=args.rtol,
            atol=args.atol,
        )

    print(f"Exported fused ONNX to: {output_path}")
    print(
        f"Architecture: obs_encoding_size={arch.obs_encoding_size}, "
        f"seq_len={arch.seq_len}, layers={arch.mha_num_attention_layers}, "
        f"ff_factor={arch.mha_ff_dim_factor}, action_shape=[{arch.action_chunk_size}, {arch.action_dim}]"
    )
    if args.verify_onnxruntime:
        print("ONNXRuntime verification: OK")


if __name__ == "__main__":
    main()
