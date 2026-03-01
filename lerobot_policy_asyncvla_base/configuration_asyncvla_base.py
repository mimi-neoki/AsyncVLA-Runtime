from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AsyncVLABasePolicyConfig:
    """Configuration for workstation-side AsyncVLA base policy."""

    snapshot_dir: str
    device: str = "cuda"
    dtype: str = "float16"
    image_key: str = "front_image"
    instruction_key: str = "instruction"
    goal_pose_key: str = "goal_pose"
    timestamp_key: str = "timestamp_ns"
    action_chunk_size: int = 8
    hidden_dim: int = 4096
    projected_dim: int = 1024
    projector_checkpoint: str = "action_proj--750000_checkpoint.pt"

    def resolve_snapshot_dir(self) -> Path:
        path = Path(self.snapshot_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"snapshot_dir not found: {path}")
        return path

    def resolve_projector_path(self) -> Path:
        return self.resolve_snapshot_dir() / self.projector_checkpoint
