#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


class EdgeAdapterExportWrapper(nn.Module):
    """Fallback export wrapper when no custom edge module is provided."""

    def __init__(self, chunk_size: int, projected_dim: int, pose_dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.pose_dim = pose_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(projected_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + 32 + 128 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * pose_dim),
        )

    def forward(
        self,
        current_image: torch.Tensor,
        delayed_image: torch.Tensor,
        projected_tokens: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        curr = self.image_encoder(current_image)
        delay = self.image_encoder(delayed_image)
        token_feat = self.token_proj(projected_tokens.mean(dim=1))
        fused = torch.cat([curr, delay, token_feat, goal_pose], dim=-1)
        out = self.head(fused)
        return out.view(out.shape[0], self.chunk_size, self.pose_dim)


def _resolve_factory(factory: str | None) -> Callable[[], nn.Module] | None:
    if not factory:
        return None
    module_name, attr_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    if not callable(value):
        raise TypeError(f"Factory is not callable: {factory}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AsyncVLA edge adapter to ONNX")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--projected-dim", type=int, default=1024)
    parser.add_argument("--pose-dim", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--model-factory",
        default=None,
        help="Optional import path module:function returning nn.Module",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    factory = _resolve_factory(args.model_factory)
    if factory is not None:
        model = factory()
    else:
        model = EdgeAdapterExportWrapper(
            chunk_size=args.chunk_size,
            projected_dim=args.projected_dim,
            pose_dim=args.pose_dim,
        )
    model.eval()

    b = args.batch_size
    current_image = torch.randn(b, 3, args.image_height, args.image_width)
    delayed_image = torch.randn(b, 3, args.image_height, args.image_width)
    projected_tokens = torch.randn(b, args.chunk_size, args.projected_dim)
    goal_pose = torch.randn(b, 3)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (current_image, delayed_image, projected_tokens, goal_pose),
        str(output_path),
        input_names=["current_image", "delayed_image", "projected_tokens", "goal_pose"],
        output_names=["action_chunk"],
        dynamic_axes={
            "current_image": {0: "batch"},
            "delayed_image": {0: "batch"},
            "projected_tokens": {0: "batch"},
            "goal_pose": {0: "batch"},
            "action_chunk": {0: "batch"},
        },
        opset_version=args.opset,
    )

    print(f"Exported ONNX to: {output_path}")


if __name__ == "__main__":
    main()
