#!/usr/bin/env python3
"""Save identical HF, ONNX and DFC inputs to diagnose conversion errors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asyncvla_pi.torch_edge_runner import TorchEdgeRunner, TorchEdgeRunnerConfig
from asyncvla_pi.token_quant import load_token_quant_params, quantize_tokens_fixed_affine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--calib-dir", default="calib_data")
    parser.add_argument("--onnx", default="build_fixed/edge_adapter_fused_fp32.onnx")
    parser.add_argument("--delayed-mode", choices=["same", "roll_fullset"], default="same")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calib = Path(args.calib_dir)
    samples = json.loads((calib / "samples.json").read_text())
    tokens = np.load(calib / "calib_projected_tokens_8x1024_float32.npy", mmap_mode="r")
    indices = np.linspace(0, len(samples) - 1, args.num_samples, dtype=int)
    params = load_token_quant_params(calib / "token_quant_fixed_affine_p99_0_uint8.npz")
    runner = TorchEdgeRunner(TorchEdgeRunnerConfig(hf_dir=args.hf_dir, device="cuda", dtype="float32"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])
    saved: dict[str, list[np.ndarray]] = {}
    layer_values: dict[str, np.ndarray] = {}

    def capture(name):
        def hook(_module, _inputs, output):
            layer_values[name] = output.detach().cpu().numpy()
        return hook

    runner.model.compress_obs_enc.register_forward_hook(capture("fc1"))
    runner.model.compress_cat_enc.register_forward_hook(capture("fc2"))
    for index, layer in enumerate(runner.model.decoder.sa_decoder.layers):
        layer.register_forward_hook(capture(f"decoder{index}"))

    def append(key: str, value: np.ndarray) -> None:
        saved.setdefault(key, []).append(value)

    for idx in indices:
        prev = idx if args.delayed_mode == "same" else (idx - 1) % len(samples)
        images = []
        for name, image_idx in [("current_image", idx), ("delayed_image", prev)]:
            path = calib / "images" / Path(samples[image_idx]["local_image_path"]).name
            rgb = cv2.resize(np.asarray(Image.open(path).convert("RGB")), (96, 96))
            img_t = runner._prep_image(rgb)
            images.append(img_t)
            append(f"native_{name}", img_t.cpu().numpy().transpose(0, 2, 3, 1))
            append(f"runtime_{name}", rgb[None])
        raw = np.array(tokens[idx:idx + 1], dtype=np.float32)
        quant = quantize_tokens_fixed_affine(raw, quant_dtype="uint8", scales=params["scales"], zero_point=int(params["zero_point"]))
        restored = (quant.astype(np.float32) - params["zero_point"]) * params["scales"]
        append("native_projected_tokens", raw[:, None])
        append("runtime_projected_tokens", quant[:, None])
        with torch.inference_mode():
            fused = runner.model.encode_fused(*images, runner._prep_tokens(raw))
            append("hf_fused", fused.cpu().numpy())
            append("hf_action", runner.model.predict_action_from_fused(fused).cpu().numpy())
            fused_q = runner.model.encode_fused(*images, runner._prep_tokens(restored))
            for name, value in layer_values.items():
                append(f"layer_{name}", value)
            append("hf_token_roundtrip_fused", fused_q.cpu().numpy())
            append("hf_token_roundtrip_action", runner.model.predict_action_from_fused(fused_q).cpu().numpy())
        ort_fused = session.run(None, {"current_image": images[0].cpu().numpy(), "delayed_image": images[1].cpu().numpy(), "projected_tokens": raw})[0]
        append("onnx_fused", ort_fused.reshape(1, -1))
        with torch.inference_mode():
            append("onnx_action", runner.model.predict_action_from_fused(torch.from_numpy(ort_fused.reshape(1, -1)).cuda()).cpu().numpy())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "inputs.npz", indices=indices, **{k: np.concatenate(v) for k, v in saved.items()})
    print(f"Saved {len(indices)} matched samples to {out}")


if __name__ == "__main__":
    main()
