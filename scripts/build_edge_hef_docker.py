#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an edge adapter HEF with Hailo DFC 5.2.0 in Docker."
    )
    parser.add_argument("--base-har", default="build_fixed/edge_adapter_base.har")
    parser.add_argument("--profile", required=True, help="Path to .alls profile")
    parser.add_argument("--calib-pack", required=True, help="Path to calibration .npz")
    parser.add_argument("--workdir", required=True, help="Temporary working directory")
    parser.add_argument("--stem", required=True, help="Output stem, e.g. edge_adapter_balanced_full1024_runtimeuint8")
    parser.add_argument("--image", default="hailo_ai_sw_suite_2026-01:1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_har = (repo_root / args.base_har).resolve()
    profile = (repo_root / args.profile).resolve()
    calib_pack = (repo_root / args.calib_pack).resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    workdir.chmod(0o777)

    for src, dst_name in (
        (base_har, "edge_adapter_base.har"),
        (profile, "model.alls"),
        (calib_pack, calib_pack.name),
    ):
        subprocess.run(["cp", "-f", str(src), str(workdir / dst_name)], check=True)

    calib_name = calib_pack.name
    stem = args.stem
    container_cmd = (
        "set -euo pipefail; "
        f"hailo optimize edge_adapter_base.har --hw-arch hailo10h --model-script model.alls "
        f"--calib-set-path {shlex.quote(calib_name)} --output-har-path {shlex.quote(stem)}_quant.har 2>&1 | tee hailo_sdk.client.log; "
        f"hailo compiler {shlex.quote(stem)}_quant.har --hw-arch hailo10h --output-dir . "
        f"--output-har-path {shlex.quote(stem)}_compiled.har 2>&1 | tee -a hailo_sdk.client.log"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir}:/tmp/work",
            "-w",
            "/tmp/work",
            args.image,
            "bash",
            "-lc",
            container_cmd,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
