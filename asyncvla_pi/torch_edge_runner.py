from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .edge_adapter_model import load_edge_adapter_from_hf_snapshot

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class TorchEdgeRunnerConfig:
    hf_dir: str
    checkpoint_name: str = "shead--750000_checkpoint.pt"
    mha_num_attention_heads: int = 4
    image_height: int = 96
    image_width: int = 96
    normalize_imagenet: bool = True
    image_scale_255: bool = True
    convert_bgr_to_rgb: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    preprocess_mode: str = "hf"
    token_uint8_mode: str = "dynamic_minmax"


class TorchEdgeRunner:
    """Runs the edge adapter directly from the HF checkpoint via PyTorch."""

    def __init__(self, config: TorchEdgeRunnerConfig) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for TorchEdgeRunner")
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        if self.device.type == "cpu" and self.dtype != torch.float32:
            raise ValueError("TorchEdgeRunner on CPU only supports dtype=float32")
        model, _, missing, unexpected = load_edge_adapter_from_hf_snapshot(
            hf_dir=Path(config.hf_dir).expanduser().resolve(),
            checkpoint_name=config.checkpoint_name,
            mha_num_attention_heads=config.mha_num_attention_heads,
            strict=True,
        )
        if missing or unexpected:
            raise RuntimeError(f"Edge adapter checkpoint mismatch. missing={missing} unexpected={unexpected}")
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

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(image, (self.config.image_width, self.config.image_height))
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, self.config.image_height).astype(np.int32)
        x_idx = np.linspace(0, w - 1, self.config.image_width).astype(np.int32)
        return image[np.ix_(y_idx, x_idx)]

    @staticmethod
    def _quantize_unsigned_dynamic_minmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        v_min = arr.min(axis=tuple(range(1, arr.ndim)), keepdims=True)
        v_max = arr.max(axis=tuple(range(1, arr.ndim)), keepdims=True)
        denom = np.maximum(v_max - v_min, 1e-6)
        scaled = (arr - v_min) / denom * 255.0
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)

    def _prep_image(self, image: np.ndarray) -> Any:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, got {arr.shape}")
        if self.config.convert_bgr_to_rgb:
            if arr.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image for BGR->RGB conversion, got {arr.shape}")
            if cv2 is not None:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                arr = arr[..., ::-1]
        resized = self._resize(arr).astype(np.float32)
        preprocess_mode = self.config.preprocess_mode.strip().lower()
        if preprocess_mode == "hf":
            if self.config.image_scale_255 and resized.max() > 1.0:
                resized = resized / 255.0
            if self.config.normalize_imagenet:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                resized = (resized - mean) / std
        elif preprocess_mode == "hailo_int8norm":
            resized = resized - 128.0
        else:
            raise ValueError(f"Unsupported preprocess_mode: {self.config.preprocess_mode}")
        chw = np.transpose(resized, (2, 0, 1))[None, ...]
        return torch.from_numpy(chw).to(device=self.device, dtype=self.dtype)

    def _prep_tokens(self, projected_tokens: np.ndarray) -> Any:
        tokens = np.asarray(projected_tokens, dtype=np.float32)
        if tokens.ndim == 2:
            tokens = tokens[None, ...]
        preprocess_mode = self.config.preprocess_mode.strip().lower()
        if preprocess_mode == "hailo_int8norm":
            if self.config.token_uint8_mode == "dynamic_minmax":
                tokens = self._quantize_unsigned_dynamic_minmax(tokens)
            else:
                tokens = np.clip(np.round(tokens), 0, 255).astype(np.uint8)
            tokens = tokens.astype(np.float32) - 128.0
        elif preprocess_mode != "hf":
            raise ValueError(f"Unsupported preprocess_mode: {self.config.preprocess_mode}")
        return torch.from_numpy(tokens).to(device=self.device, dtype=self.dtype)

    def infer(
        self,
        current_image: np.ndarray,
        delayed_image: np.ndarray,
        projected_tokens: np.ndarray,
        goal_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        del goal_pose
        current_t = self._prep_image(current_image)
        delayed_t = self._prep_image(delayed_image)
        tokens_t = self._prep_tokens(projected_tokens)
        with torch.inference_mode():
            out = self.model(current_t, delayed_t, tokens_t)
        return out.detach().cpu().to(torch.float32).numpy()

    def close(self) -> None:
        return
