#!/usr/bin/env python3
"""Run a saved HAR stage in the DFC container on matched diagnostic inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--stage", choices=["native", "fp", "quantized", "bit_exact"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--custom-infer-config", help="DFC YAML/JSON for diagnostic partial quantization")
    parser.add_argument("--layers", nargs="+", help="Save selected internal outputs as an NPZ (DFC 5.2 API)")
    parser.add_argument("--bit-exact-layernorm", action="store_true", help="Diagnostic DFC 5.2 internal LayerNorm emulation; not a build setting")
    args = parser.parse_args()
    runner = ClientRunner(har=args.har, hw_arch="hailo10h")
    hn = runner.get_hn_dict()
    source = np.load(args.inputs, allow_pickle=False)
    prefix = "native" if args.stage == "native" else "runtime"
    data = {}
    for name, layer in hn["layers"].items():
        if layer["type"] != "input_layer":
            continue
        original = layer["original_names"][0]
        data[name] = source[f"{prefix}_{original}"].astype(np.float32)
    mode = {
        "native": InferenceContext.SDK_NATIVE,
        "fp": InferenceContext.SDK_FP_OPTIMIZED,
        "quantized": InferenceContext.SDK_QUANTIZED,
        "bit_exact": InferenceContext.SDK_BIT_EXACT,
    }[args.stage]
    print(json.dumps({"stage": args.stage, "inputs": {k: list(v.shape) for k, v in data.items()}}), flush=True)
    with runner.infer_context(mode, custom_infer_config=args.custom_infer_config) as ctx:
        if args.bit_exact_layernorm:
            if args.stage != "quantized" or args.layers:
                raise ValueError("--bit-exact-layernorm requires quantized stage without --layers")
            import tensorflow as tf

            model = runner.get_keras_model(ctx)
            model.build({name: values[:1] for name, values in data.items()})
            changed = []
            for name, layer in model.model.layers.items():
                if "layer_normalization" in name:
                    for op in layer.atomic_ops:
                        op.bit_exact = True
                    changed.append(name)
            if not changed:
                raise RuntimeError("No decomposed LayerNorm operators found")
            print(json.dumps({"bit_exact_layernorm_layers": changed}), flush=True)
            output = model.run(tf.data.Dataset.from_tensor_slices(data), batch_size=1, data_count=len(next(iter(data.values()))))
            np.save(args.output, np.asarray(output))
            return
        if args.layers:
            import tensorflow as tf

            model = runner.get_keras_model(ctx)
            if args.stage == "bit_exact":
                # DFC 5.2 get_keras_model defaults this context to native mode,
                # unlike runner.infer. Mirror acceleras_inference explicitly.
                model.set_quantized()
                model.set_native(False)
                model.set_bit_exact(True)
            model.build({name: values[:1] for name, values in data.items()})
            model.model.set_output_interal_layers(args.layers)

            @tf.function
            def run(batch):
                return model(batch, training=False)

            collected = {}
            for index in range(len(next(iter(data.values())))):
                outputs, internal = run({k: tf.convert_to_tensor(v[index:index + 1]) for k, v in data.items()})
                collected.setdefault("output", []).append(np.asarray(outputs))
                for name, tensors in zip(args.layers, internal):
                    tensors = tensors if isinstance(tensors, (tuple, list)) else [tensors]
                    for i, tensor in enumerate(tensors):
                        collected.setdefault(f"{name}:{i}", []).append(np.asarray(tensor))
            np.savez(args.output, **{k: np.concatenate(v) for k, v in collected.items()})
            print(f"Saved intermediate outputs to {args.output}", flush=True)
            return
        output = runner.infer(ctx, data, batch_size=1)
    output = np.asarray(output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(json.dumps({"output": args.output, "shape": list(output.shape), "min": float(output.min()), "max": float(output.max())}), flush=True)


if __name__ == "__main__":
    main()
