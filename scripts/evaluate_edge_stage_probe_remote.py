#!/usr/bin/env python3
"""Evaluate HEFs on the exact images/tokens saved for DFC stage emulation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_remote_edge_hef_candidates import (
    _build_payload,
    _compare_outputs,
    _post_infer,
    _summarize_candidate,
    _upload_hef,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--hef", action="append", required=True)
    parser.add_argument("--edge-url", default="http://openduck.local:8100/infer")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--save-json", required=True)
    args = parser.parse_args()
    data = np.load(args.inputs, allow_pickle=False)
    destination = Path(args.save_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"inputs": str(Path(args.inputs).resolve()), "reference": "saved HF float32", "image_encoding": "raw", "candidates": []}
    for path in args.hef:
        hef = Path(path).resolve()
        uploaded = _upload_hef(args.edge_url, hef, args.timeout_s)
        candidate = {"hef": str(hef), "upload": uploaded, "samples": []}
        report["candidates"].append(candidate)
        for pos, index in enumerate(data["indices"]):
            payload = _build_payload(
                data["runtime_current_image"][pos],
                data["runtime_delayed_image"][pos],
                data["native_projected_tokens"][pos, 0],
                jpeg_quality=90,
                image_encoding="raw",
            )
            payload["include_diagnostics"] = True
            if pos == 0:
                for _ in range(max(0, args.warmup)):
                    _post_infer(args.edge_url, args.timeout_s, payload)
            response = _post_infer(args.edge_url, args.timeout_s, payload)
            reference = data["hf_action"][pos:pos + 1]
            result = {
                "sample_index": int(index),
                "diff_report": _compare_outputs(reference, np.asarray(response["hef_action_chunk"])),
                "server_reference_diff": _compare_outputs(reference, np.asarray(response["torch_action_chunk"])),
                "latency_ms": response["latency_ms"],
                "hef_action_chunk": response["hef_action_chunk"],
                "reference_action_chunk": reference.tolist(),
                "diagnostics": response.get("diagnostics"),
            }
            diagnostics = result["diagnostics"] or {}
            if "hef_fused_feature" in diagnostics:
                result["fused_diff"] = _compare_outputs(data["hf_fused"][pos:pos + 1], np.asarray(diagnostics["hef_fused_feature"]))
            candidate["samples"].append(result)
            candidate["summary"] = _summarize_candidate(candidate["samples"])
            # Retain completed requests if the device or server fails mid-evaluation.
            destination.write_text(json.dumps(report, indent=2) + "\n")
            if (pos + 1) % 16 == 0:
                print(f"{hef.name}: {pos + 1}/{len(data['indices'])}", flush=True)
        print(json.dumps(candidate["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
