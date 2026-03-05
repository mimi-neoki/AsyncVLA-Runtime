from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AsyncVLABasePolicyConfig:
    """Configuration for workstation-side AsyncVLA base policy."""

    snapshot_dir: str
    device: str = "cuda"
    dtype: str = "float16"
    quantization: str = "none"
    image_key: str = "front_image"
    instruction_key: str = "instruction"
    goal_pose_key: str = "goal_pose"
    goal_image_key: str = "goal_image"
    timestamp_key: str = "timestamp_ns"
    action_chunk_size: int = 8
    num_images_in_input: int = 2
    hidden_dim: int = 4096
    projected_dim: int = 1024
    unnorm_key: str | None = None
    task_id: int | None = None
    task_mode: str = "auto"
    task_mode_key: str = "task_mode"
    task_id_key: str = "task_id"
    satellite_key: str = "satellite"
    satellite_default: bool = False
    asyncvla_repo_dir: str = "~/gitrepo/AsyncVLA"
    task_id_image_only: int = 6
    task_id_pose_only: int = 4
    task_id_instruction_only: int = 7
    task_id_instruction_with_pose: int = 8
    duplicate_current_image_as_goal: bool = True
    projector_checkpoint: str = "action_proj--750000_checkpoint.pt"
    pose_projector_checkpoint: str = "pose_projector--750000_checkpoint.pt"

    def resolve_snapshot_dir(self) -> Path:
        path = Path(self.snapshot_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"snapshot_dir not found: {path}")
        return path

    def resolve_projector_path(self) -> Path:
        return self.resolve_snapshot_dir() / self.projector_checkpoint

    def resolve_pose_projector_path(self) -> Path:
        return self.resolve_snapshot_dir() / self.pose_projector_checkpoint

    def resolve_asyncvla_repo_dir(self) -> Path:
        return Path(self.asyncvla_repo_dir).expanduser().resolve()
