from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .configuration_asyncvla_base import AsyncVLABasePolicyConfig

try:
    from lerobot.common.policies.pretrained import PreTrainedPolicy
except Exception:  # pragma: no cover
    class PreTrainedPolicy(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()


@dataclass
class GuidancePacket:
    projected_tokens: np.ndarray
    timestamp_ns: int


class _BaseStub(nn.Module):
    """Lightweight stand-in for base VLA execution wiring.

    Replace this module with the actual AsyncVLA base model call in production.
    """

    def __init__(self, hidden_dim: int, action_chunk_size: int) -> None:
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image: torch.Tensor, instruction_bias: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(-1, -2))  # [B, 3]
        hidden = self.proj(pooled)
        hidden = hidden + instruction_bias
        return hidden.unsqueeze(1).repeat(1, self.action_chunk_size, 1)


class AsyncVLABasePolicy(PreTrainedPolicy):
    """Workstation-side policy that returns projected guidance tokens.

    This class is intentionally shaped for LeRobot PolicyServer usage.
    The returned packet contains projected action-token embeddings and
    the timestamp echoed from the observation.
    """

    def __init__(self, config: AsyncVLABasePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.dtype, torch.float16)
        self.base_model = _BaseStub(config.hidden_dim, config.action_chunk_size)
        self.action_proj = nn.Linear(config.hidden_dim, config.projected_dim)
        self.to(self.device)
        self.eval()
        self._load_projector_checkpoint()

    @classmethod
    def from_snapshot(cls, snapshot_dir: str, **kwargs: Any) -> "AsyncVLABasePolicy":
        cfg = AsyncVLABasePolicyConfig(snapshot_dir=snapshot_dir, **kwargs)
        return cls(cfg)

    def _load_projector_checkpoint(self) -> None:
        projector_path = self.config.resolve_projector_path()
        if not projector_path.exists():
            return
        payload = torch.load(projector_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if isinstance(payload, dict):
            cleaned: dict[str, torch.Tensor] = {}
            for key, value in payload.items():
                normalized = key.replace("module.", "")
                if normalized.startswith("action_proj."):
                    normalized = normalized[len("action_proj.") :]
                if normalized in {"weight", "bias"}:
                    cleaned[normalized] = value
            if cleaned:
                self.action_proj.load_state_dict(cleaned, strict=False)

    def _extract_image(self, observation: dict[str, Any]) -> np.ndarray:
        value = observation.get(self.config.image_key)
        if value is None and isinstance(observation.get("images"), dict):
            value = observation["images"].get(self.config.image_key)
        if value is None:
            raise KeyError(f"Missing image key: {self.config.image_key}")
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape: {array.shape}")
        return array

    def _instruction_bias(self, observation: dict[str, Any]) -> torch.Tensor:
        instruction = str(observation.get(self.config.instruction_key, ""))
        goal_pose = np.asarray(observation.get(self.config.goal_pose_key, [0.0, 0.0, 0.0]), dtype=np.float32)
        seed = sum(ord(ch) for ch in instruction) % 10000
        torch.manual_seed(seed)
        bias = torch.randn((1, self.config.hidden_dim), dtype=torch.float32)
        if goal_pose.size >= 2:
            pose_scalar = float(goal_pose[:2].sum())
            bias = bias + pose_scalar
        return bias.to(self.device)

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> GuidancePacket:
        image = self._extract_image(observation)
        image_t = torch.from_numpy(image).to(self.device)
        image_t = image_t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        instruction_bias = self._instruction_bias(observation)
        tokens = self.base_model(image_t, instruction_bias)
        projected = self.action_proj(tokens)
        projected_np = projected.squeeze(0).detach().cpu().float().numpy()
        timestamp_ns = int(observation.get(self.config.timestamp_key, time.monotonic_ns()))
        return GuidancePacket(projected_tokens=projected_np, timestamp_ns=timestamp_ns)

    @torch.inference_mode()
    def forward(self, observation: dict[str, Any]) -> dict[str, Any]:
        packet = self.infer(observation)
        return {
            "projected_tokens": packet.projected_tokens,
            "timestamp_ns": packet.timestamp_ns,
        }

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.forward(observation)


def validate_snapshot_layout(snapshot_dir: str | Path) -> dict[str, bool]:
    """Small helper to verify expected HF snapshot members."""

    root = Path(snapshot_dir)
    expected = [
        "config.json",
        "model.safetensors.index.json",
        "lora_adapter",
        "action_proj--750000_checkpoint.pt",
    ]
    return {item: (root / item).exists() for item in expected}
