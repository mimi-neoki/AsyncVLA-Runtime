from __future__ import annotations

import numpy as np
import torch

from asyncvla_pi.torch_edge_runner import TorchEdgeRunner, TorchEdgeRunnerConfig


class _DummyEdgeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_obs: torch.Tensor | None = None
        self.last_past: torch.Tensor | None = None
        self.last_tokens: torch.Tensor | None = None

    def forward(self, obs_img: torch.Tensor, past_img: torch.Tensor, vla_feature: torch.Tensor) -> torch.Tensor:
        self.last_obs = obs_img.detach().clone()
        self.last_past = past_img.detach().clone()
        self.last_tokens = vla_feature.detach().clone()
        batch = int(obs_img.shape[0])
        return torch.zeros((batch, 8, 4), dtype=obs_img.dtype, device=obs_img.device)


def test_torch_edge_runner_infer(monkeypatch) -> None:
    dummy = _DummyEdgeModel()

    def _fake_loader(**kwargs):
        return dummy, None, [], []

    monkeypatch.setattr("asyncvla_pi.torch_edge_runner.load_edge_adapter_from_hf_snapshot", _fake_loader)

    runner = TorchEdgeRunner(
        TorchEdgeRunnerConfig(
            hf_dir="~/dummy",
            device="cpu",
            dtype="float32",
            convert_bgr_to_rgb=True,
        )
    )
    current = np.full((120, 160, 3), 255, dtype=np.uint8)
    delayed = np.zeros((120, 160, 3), dtype=np.uint8)
    tokens = np.ones((8, 1024), dtype=np.float32)
    out = runner.infer(current_image=current, delayed_image=delayed, projected_tokens=tokens)

    assert out.shape == (1, 8, 4)
    assert dummy.last_obs is not None
    assert dummy.last_past is not None
    assert dummy.last_tokens is not None
    assert tuple(dummy.last_obs.shape) == (1, 3, 96, 96)
    assert tuple(dummy.last_past.shape) == (1, 3, 96, 96)
    assert tuple(dummy.last_tokens.shape) == (1, 8, 1024)


def test_torch_edge_runner_rejects_half_on_cpu(monkeypatch) -> None:
    def _fake_loader(**kwargs):
        return _DummyEdgeModel(), None, [], []

    monkeypatch.setattr("asyncvla_pi.torch_edge_runner.load_edge_adapter_from_hf_snapshot", _fake_loader)

    try:
        TorchEdgeRunner(TorchEdgeRunnerConfig(hf_dir="~/dummy", device="cpu", dtype="float16"))
    except ValueError as exc:
        assert "CPU only supports dtype=float32" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
