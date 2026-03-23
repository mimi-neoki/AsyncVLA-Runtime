from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .edge_adapter_model import load_edge_adapter_from_hf_snapshot
from .hailo_edge_runner import HailoEdgeRunner, HailoEdgeRunnerConfig

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class HybridEdgeRunnerConfig:
    hef_path: str
    hf_dir: str
    checkpoint_name: str = "shead--750000_checkpoint.pt"
    mha_num_attention_heads: int = 4
    input_current_image: str = "current_image"
    input_delayed_image: str = "delayed_image"
    input_projected_tokens: str = "projected_tokens"
    output_fused_feature: str = "fused_feature"
    image_height: int = 96
    image_width: int = 96
    fused_dim: int = 1024
    normalize_imagenet: bool = False
    image_layout: str = "nhwc"
    input_format_type: str = "uint8"
    output_format_type: str = "float32"
    image_scale_255: bool = False
    convert_bgr_to_rgb: bool = False
    token_uint8_mode: str = "fixed_affine"
    token_quant_params_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"


class HybridEdgeRunner:
    """Runs HEF up to fused decoder latent, then Torch action head locally."""

    def __init__(self, config: HybridEdgeRunnerConfig, target: Any | None = None) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for HybridEdgeRunner")
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        if self.device.type == "cpu" and self.dtype != torch.float32:
            raise ValueError("HybridEdgeRunner on CPU only supports dtype=float32")

        self.hailo_runner = HailoEdgeRunner(
            HailoEdgeRunnerConfig(
                hef_path=str(Path(config.hef_path).expanduser().resolve()),
                input_current_image=config.input_current_image,
                input_delayed_image=config.input_delayed_image,
                input_projected_tokens=config.input_projected_tokens,
                input_goal_pose=None,
                output_action_chunk=config.output_fused_feature,
                image_height=config.image_height,
                image_width=config.image_width,
                chunk_size=1,
                pose_dim=config.fused_dim,
                normalize_imagenet=config.normalize_imagenet,
                image_layout=config.image_layout,
                input_format_type=config.input_format_type,
                output_format_type=config.output_format_type,
                image_scale_255=config.image_scale_255,
                convert_bgr_to_rgb=config.convert_bgr_to_rgb,
                token_uint8_mode=config.token_uint8_mode,
                token_quant_params_path=config.token_quant_params_path,
            ),
            target=target,
        )

        model, arch, missing, unexpected = load_edge_adapter_from_hf_snapshot(
            hf_dir=Path(config.hf_dir).expanduser().resolve(),
            checkpoint_name=config.checkpoint_name,
            mha_num_attention_heads=config.mha_num_attention_heads,
            strict=True,
        )
        if missing or unexpected:
            raise RuntimeError(f"Edge adapter checkpoint mismatch. missing={missing} unexpected={unexpected}")
        if int(arch.obs_encoding_size) != int(config.fused_dim):
            raise ValueError(
                f"Hybrid fused_dim mismatch: config={config.fused_dim}, checkpoint={arch.obs_encoding_size}"
            )
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> Any:
        name = dtype_name.strip().lower()
        if name == "float32":
            return torch.float32
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    def infer(
        self,
        current_image: np.ndarray,
        delayed_image: np.ndarray,
        projected_tokens: np.ndarray,
        goal_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        del goal_pose
        fused = np.asarray(
            self.hailo_runner.infer(
                current_image=current_image,
                delayed_image=delayed_image,
                projected_tokens=projected_tokens,
                goal_pose=None,
            ),
            dtype=np.float32,
        )
        if fused.ndim == 3 and fused.shape[1] == 1:
            fused = fused.reshape(fused.shape[0], fused.shape[2])
        elif fused.ndim == 2:
            pass
        else:
            fused = fused.reshape(fused.shape[0], -1)
        with torch.inference_mode():
            fused_t = torch.from_numpy(fused).to(device=self.device, dtype=self.dtype)
            out = self.model.predict_action_from_fused(fused_t)
        return out.detach().cpu().to(torch.float32).numpy()

    def close(self) -> None:
        self.hailo_runner.close()

