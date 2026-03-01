from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class HailoEdgeRunnerConfig:
    hef_path: str
    input_current_image: str = "current_image"
    input_delayed_image: str = "delayed_image"
    input_projected_tokens: str = "projected_tokens"
    input_goal_pose: str = "goal_pose"
    output_action_chunk: str = "action_chunk"
    image_height: int = 224
    image_width: int = 224
    chunk_size: int = 8
    pose_dim: int = 4


class HailoEdgeRunner:
    """Runs edge adapter inference from HEF on the Pi side.

    If pyhailort is unavailable, you can pass `fallback_fn` for dry runs.
    """

    def __init__(
        self,
        config: HailoEdgeRunnerConfig,
        fallback_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.fallback_fn = fallback_fn
        self._ready = False
        self._network_group: Any = None
        self._network_group_params: Any = None
        self._infer_pipeline: Any = None

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(image, (self.config.image_width, self.config.image_height))
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, self.config.image_height).astype(np.int32)
        x_idx = np.linspace(0, w - 1, self.config.image_width).astype(np.int32)
        return image[np.ix_(y_idx, x_idx)]

    def _prep_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, got {arr.shape}")
        resized = self._resize(arr)
        return resized.astype(np.float32)[None, ...] / 255.0

    def _build_inputs(
        self,
        current_image: np.ndarray,
        delayed_image: np.ndarray,
        projected_tokens: np.ndarray,
        goal_pose: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        goal = np.asarray(goal_pose if goal_pose is not None else np.zeros((3,), dtype=np.float32), dtype=np.float32)
        goal = goal.reshape(1, -1)
        tokens = np.asarray(projected_tokens, dtype=np.float32)
        if tokens.ndim == 2:
            tokens = tokens[None, ...]
        return {
            self.config.input_current_image: self._prep_image(current_image),
            self.config.input_delayed_image: self._prep_image(delayed_image),
            self.config.input_projected_tokens: tokens,
            self.config.input_goal_pose: goal,
        }

    def _init_hailo(self) -> None:
        if self._ready:
            return
        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pyhailort is unavailable. Install hailo_platform or pass fallback_fn for simulation."
            ) from exc

        hef = HEF(str(Path(self.config.hef_path).expanduser().resolve()))
        target = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        infer_pipeline = InferVStreams(network_group, input_params, output_params)

        self._network_group = network_group
        self._network_group_params = network_group_params
        self._infer_pipeline = infer_pipeline
        self._ready = True

    def infer(
        self,
        current_image: np.ndarray,
        delayed_image: np.ndarray,
        projected_tokens: np.ndarray,
        goal_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        inputs = self._build_inputs(current_image, delayed_image, projected_tokens, goal_pose)

        if self.fallback_fn is not None:
            out = np.asarray(self.fallback_fn(inputs), dtype=np.float32)
            if out.ndim == 2:
                out = out[None, ...]
            return out

        self._init_hailo()
        with self._network_group.activate(self._network_group_params):
            output_dict = self._infer_pipeline.infer(inputs)
        if self.config.output_action_chunk in output_dict:
            output = output_dict[self.config.output_action_chunk]
        else:
            output = next(iter(output_dict.values()))
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 2:
            output = output[None, ...]
        if output.ndim != 3:
            output = output.reshape(1, self.config.chunk_size, self.config.pose_dim)
        return output
