from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from asyncvla_pi.edge_adapter_model import (
    infer_edge_adapter_architecture,
    load_edge_adapter_from_hf_snapshot,
    load_torch_state_dict,
)


DEFAULT_HF_DIR = Path(os.environ.get("ASYNCVLA_HF_DIR", "/home/mimi/gitrepo/AsyncVLA_release"))


@pytest.fixture(scope="module")
def hf_dir() -> Path:
    if not DEFAULT_HF_DIR.exists():
        pytest.skip(f"HF snapshot not found: {DEFAULT_HF_DIR}")
    return DEFAULT_HF_DIR


def test_infer_architecture_from_shead(hf_dir: Path) -> None:
    state_dict = load_torch_state_dict(hf_dir / "shead--750000_checkpoint.pt")
    arch = infer_edge_adapter_architecture(state_dict)
    assert arch.obs_encoding_size == 1024
    assert arch.seq_len == 10
    assert arch.mha_num_attention_layers == 4
    assert arch.mha_ff_dim_factor == 4
    assert arch.action_chunk_size == 8
    assert arch.action_dim == 4


def test_load_and_forward_edge_adapter(hf_dir: Path) -> None:
    model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
        hf_dir=hf_dir,
        checkpoint_name="shead--750000_checkpoint.pt",
        mha_num_attention_heads=4,
        strict=True,
    )
    assert missing == []
    assert unexpected == []

    obs = torch.randn(1, 3, 96, 96)
    past = torch.randn(1, 3, 96, 96)
    projected = torch.randn(1, arch.action_chunk_size, arch.obs_encoding_size)
    with torch.inference_mode():
        out = model(obs, past, projected)

    assert tuple(out.shape) == (1, 8, 4)
    assert torch.isfinite(out).all()


def test_export_cli_with_onnxruntime_verification(hf_dir: Path, tmp_path: Path) -> None:
    onnx_path = tmp_path / "edge_adapter.onnx"
    cmd = [
        sys.executable,
        "export/export_edge_to_onnx.py",
        "--hf-dir",
        str(hf_dir),
        "--output",
        str(onnx_path),
        "--verify-onnxruntime",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"export command failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0
