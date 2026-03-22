from __future__ import annotations

from pathlib import Path

import numpy as np


def load_token_quant_params(path: str | Path) -> dict[str, np.ndarray | float | int]:
    resolved = Path(path).expanduser().resolve()
    with np.load(resolved) as data:
        scales = np.asarray(data["scales"], dtype=np.float32)
        zero_point = int(data.get("zero_point", 128))
    if scales.ndim != 1:
        raise ValueError(f"Expected 1D token scales, got shape={scales.shape}")
    if np.any(scales <= 0):
        raise ValueError("Token scales must be positive")
    return {
        "path": str(resolved),
        "scales": scales,
        "zero_point": zero_point,
    }


def quantize_tokens_fixed_affine(
    values: np.ndarray,
    *,
    quant_dtype: str,
    scales: np.ndarray,
    zero_point: int = 128,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected token array with ndim>=2, got shape={arr.shape}")
    if arr.shape[-1] != int(scales.shape[0]):
        raise ValueError(
            f"Token feature size mismatch: values.shape[-1]={arr.shape[-1]} scales={scales.shape[0]}"
        )
    scaled = arr / scales.reshape(*([1] * (arr.ndim - 1)), -1)
    if quant_dtype == "int8":
        return np.clip(np.round(scaled), -128, 127).astype(np.int8)
    return np.clip(np.round(scaled + float(zero_point)), 0, 255).astype(np.uint8)


def build_token_quant_params(
    projected_tokens: np.ndarray,
    *,
    percentile: float = 100.0,
    quant_dtype: str = "uint8",
) -> dict[str, np.ndarray | float | int]:
    arr = np.asarray(projected_tokens, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected projected tokens with shape [N, 8, 1024], got {arr.shape}")
    flat = np.abs(arr).reshape(-1, arr.shape[-1])
    if percentile >= 100.0:
        max_abs = flat.max(axis=0)
    else:
        max_abs = np.percentile(flat, percentile, axis=0)
    denom = 127.0 if quant_dtype == "int8" else 127.0
    scales = np.maximum(max_abs / denom, 1e-6).astype(np.float32)
    return {
        "scales": scales,
        "zero_point": 128 if quant_dtype == "uint8" else 0,
        "percentile": float(percentile),
    }
