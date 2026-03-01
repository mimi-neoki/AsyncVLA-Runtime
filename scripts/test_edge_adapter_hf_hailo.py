#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import torch
import yaml


def remove_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def infer_action_head_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    fc1_w = state_dict["model.fc1.weight"]
    fc2_w = state_dict["model.fc2.weight"]
    hidden_dim = int(fc1_w.shape[0])
    in_features = int(fc1_w.shape[1])
    input_dim = (in_features - 1) // 4
    action_dim = int(fc2_w.shape[0])
    return input_dim, hidden_dim, action_dim


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="Load AsyncVLA edge-side weights from HF snapshot and test inference")
    parser.add_argument(
        "--asyncvla-root",
        default="/home/pi/gitrepo/AsyncVLA",
        help="Path to AsyncVLA source repo",
    )
    parser.add_argument(
        "--hf-dir",
        default="/home/pi/gitrepo/AsyncVLA/AsyncVLA_release",
        help="Path to HF snapshot directory",
    )
    parser.add_argument(
        "--h10-hef",
        default="/usr/share/hailo-models/resnet_v1_50_h10.hef",
        help="HEF path for Hailo-10H runtime test (must be H10 architecture)",
    )
    parser.add_argument("--benchmark-seconds", type=int, default=3)
    parser.add_argument("--skip-hailo", action="store_true")
    args = parser.parse_args()

    asyncvla_root = Path(args.asyncvla_root).expanduser().resolve()
    hf_dir = Path(args.hf_dir).expanduser().resolve()
    h10_hef = Path(args.h10_hef).expanduser().resolve()

    if not asyncvla_root.exists():
        raise FileNotFoundError(f"AsyncVLA root not found: {asyncvla_root}")
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF snapshot not found: {hf_dir}")

    # Import AsyncVLA modules from source tree without executing prismatic/__init__.py.
    sys.path.insert(0, str(asyncvla_root))
    prismatic_dir = asyncvla_root / "prismatic"
    pkg = types.ModuleType("prismatic")
    pkg.__path__ = [str(prismatic_dir)]  # type: ignore[attr-defined]
    sys.modules["prismatic"] = pkg
    pkg_models = types.ModuleType("prismatic.models")
    pkg_models.__path__ = [str(prismatic_dir / "models")]  # type: ignore[attr-defined]
    sys.modules["prismatic.models"] = pkg_models
    pkg_vla = types.ModuleType("prismatic.vla")
    pkg_vla.__path__ = [str(prismatic_dir / "vla")]  # type: ignore[attr-defined]
    sys.modules["prismatic.vla"] = pkg_vla

    from prismatic.models.action_heads import L1RegressionActionHead_idcat
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.small_head import Edge_adapter

    cfg_path = asyncvla_root / "config_nav" / "dataset_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    obs_encoding_size = int(cfg["obs_encoding_size"])
    nhead = int(cfg["mha_num_attention_heads"])
    nlayers = int(cfg["mha_num_attention_layers"])
    ff_factor = int(cfg["mha_ff_dim_factor"])

    # 1) Load edge adapter (shead)
    shead_ckpt = hf_dir / "shead--750000_checkpoint.pt"
    shead_state = remove_module_prefix(torch.load(shead_ckpt, map_location="cpu"))
    shead = Edge_adapter(
        obs_encoding_size=obs_encoding_size,
        mha_num_attention_heads=nhead,
        mha_num_attention_layers=nlayers,
        mha_ff_dim_factor=ff_factor,
    )
    missing_s, unexpected_s = shead.load_state_dict(shead_state, strict=True)

    # 2) Load pose projector checkpoint (edge-side dependency in AsyncVLA split setup)
    pose_ckpt = hf_dir / "pose_projector--750000_checkpoint.pt"
    pose_state = remove_module_prefix(torch.load(pose_ckpt, map_location="cpu"))
    llm_dim = int(pose_state["fc1.weight"].shape[0])
    proprio_dim = int(pose_state["fc1.weight"].shape[1])
    pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim)
    missing_p, unexpected_p = pose_projector.load_state_dict(pose_state, strict=True)

    # 3) Load action head checkpoint for consistency check
    action_ckpt = hf_dir / "action_head--750000_checkpoint.pt"
    action_state = remove_module_prefix(torch.load(action_ckpt, map_location="cpu"))
    in_dim, hidden_dim, action_dim = infer_action_head_dims(action_state)
    action_head = L1RegressionActionHead_idcat(
        input_dim=in_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )
    missing_a, unexpected_a = action_head.load_state_dict(action_state, strict=True)

    shead.eval()
    pose_projector.eval()
    action_head.eval()

    # Edge-only inference test (projected tokens are assumed to come from remote base VLA)
    torch.manual_seed(7)
    obs_img = torch.randn(1, 3, 96, 96)
    past_img = torch.randn(1, 3, 96, 96)
    projected_tokens = torch.randn(1, 8, obs_encoding_size)

    with torch.inference_mode():
        delta_actions = shead(obs_img, past_img, projected_tokens)

    print("[Edge weights load] OK")
    print(f"shead checkpoint: {shead_ckpt}")
    print(f"pose projector checkpoint: {pose_ckpt}")
    print(f"action head checkpoint: {action_ckpt}")
    print(f"shead missing/unexpected: {len(missing_s)}/{len(unexpected_s)}")
    print(f"pose missing/unexpected: {len(missing_p)}/{len(unexpected_p)}")
    print(f"action missing/unexpected: {len(missing_a)}/{len(unexpected_a)}")
    print(f"delta_actions shape: {tuple(delta_actions.shape)}")
    print(
        "delta_actions stats:",
        {
            "min": float(delta_actions.min()),
            "max": float(delta_actions.max()),
            "mean": float(delta_actions.mean()),
        },
    )

    if args.skip_hailo:
        return 0

    if not h10_hef.exists():
        raise FileNotFoundError(
            f"H10 HEF not found: {h10_hef}. Provide --h10-hef to run hardware inference check."
        )

    # Hailo runtime smoke test on Hailo-10H.
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
    # Show just tail metrics from benchmark output for readability.
    bench_lines = [line for line in bench.stdout.strip().splitlines() if line.strip()]
    print("benchmark tail:")
    for line in bench_lines[-12:]:
        print(line)

    print(
        "\nNOTE: Hailo execution above validates Hailo-10H runtime path. "
        "Running AsyncVLA edge adapter on Hailo requires an edge-adapter HEF compiled offline (x86 Dataflow Compiler)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
