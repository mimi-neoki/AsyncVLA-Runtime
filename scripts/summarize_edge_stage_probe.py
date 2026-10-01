#!/usr/bin/env python3
"""Compare fused HAR stages and their identical Torch action heads against HF."""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asyncvla_pi.torch_edge_runner import TorchEdgeRunner, TorchEdgeRunnerConfig


def metrics(reference: np.ndarray, values: np.ndarray) -> dict[str, float]:
    a = reference.reshape(len(reference), -1).astype(np.float64)
    b = values.reshape(len(values), -1).astype(np.float64)
    diff = a - b
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return {
        "rmse_global": float(np.sqrt(np.mean(diff**2))),
        "rmse_mean": float(np.sqrt(np.mean(diff**2, axis=1)).mean()),
        "mean_abs": float(np.abs(diff).mean()),
        "max_abs": float(np.abs(diff).max()),
        "cosine_mean": float(np.mean(np.sum(a*b, axis=1) / np.maximum(denom, 1e-20))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--quant-har", default="build_fixed/edge_adapter_fused_hfimg_fixedp99_0_full1024_quant.har")
    args = parser.parse_args()
    directory = Path(args.probe_dir)
    data = np.load(directory / "inputs.npz")
    runner = TorchEdgeRunner(TorchEdgeRunnerConfig(hf_dir=args.hf_dir, device="cuda"))
    with tarfile.open(args.quant_har) as archive:
        with np.load(io.BytesIO(archive.extractfile("model.q.npz").read())) as params:
            zero_point, scale = params["model/slice1/qp_out:0"]
            layer_qparams = {name: params[f"model/{name}/qp_out:0"] for name in ["fc1", "fc2", "conv134", "conv138", "conv142", "conv146"]}
    fused = data["hf_fused"]
    qdq = (np.clip(np.round(fused / scale + zero_point), 0, 255) - zero_point) * scale
    stages = {"onnx": data["onnx_fused"], "token_roundtrip": data["hf_token_roundtrip_fused"], "output_qdq_only": qdq}
    for stage in ["native", "fp", "quantized"]:
        path = directory / f"{stage}.npy"
        if path.exists():
            stages[stage] = np.load(path).reshape(len(fused), -1)
    report = {"samples": len(fused), "sample_indices": data["indices"].tolist(), "output_quant": {"scale": float(scale), "zero_point": float(zero_point)}, "stages": {}}
    for name, values in stages.items():
        with torch.inference_mode():
            actions = runner.model.predict_action_from_fused(torch.from_numpy(values.copy()).cuda()).cpu().numpy()
        np.save(directory / f"{name}_action.npy", actions)
        report["stages"][name] = {"fused_vs_hf": metrics(fused, values), "action_vs_hf": metrics(data["hf_action"], actions)}
        if name == "fp":
            report["stages"][name]["fused_vs_token_roundtrip"] = metrics(data["hf_token_roundtrip_fused"], values)
    layer_path = directory / "layers_quantized.npz"
    if layer_path.exists() and "layer_fc1" in data:
        report["layers"] = {}
        with np.load(layer_path) as internal:
            for name, ref in [("fc1", "fc1"), ("fc2", "fc2"), ("conv134", "decoder0"), ("conv138", "decoder1"), ("conv142", "decoder2"), ("conv146", "decoder3")]:
                reference = data[f"layer_{ref}"]
                zp, step = layer_qparams[name]
                # DFC internal outputs are quantized codes even when stored as float32.
                values = (internal[f"model/{name}:0"] - zp) * step
                report["layers"][name] = metrics(reference, values)
    (directory / "stage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
