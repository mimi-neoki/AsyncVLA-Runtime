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
    input_goal_pose: str | None = None
    output_action_chunk: str = "action_chunk"
    image_height: int = 96
    image_width: int = 96
    chunk_size: int = 8
    pose_dim: int = 4
    normalize_imagenet: bool = True
    image_layout: str = "nchw"
    input_format_type: str = "float32"
    output_format_type: str = "float32"
    image_scale_255: bool = True
    token_uint8_mode: str = "dynamic_minmax"


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
        self._target: Any = None
        self._infer_model: Any = None
        self._configured_infer_model: Any = None
        self._mode: str = "vstreams"

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
        if self.config.input_format_type.lower() in {"uint8", "auto"}:
            data = resized.astype(np.uint8)
        else:
            data = resized.astype(np.float32)
        if self.config.image_scale_255 and data.dtype != np.uint8 and data.max() > 1.0:
            data = data / 255.0
        if self.config.normalize_imagenet and data.dtype != np.uint8:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            data = (data - mean) / std
        if self.config.image_layout.lower() == "nchw":
            data = np.transpose(data, (2, 0, 1))
        elif self.config.image_layout.lower() != "nhwc":
            raise ValueError(f"Unsupported image_layout: {self.config.image_layout}")
        return data[None, ...]

    def _prep_tokens(self, projected_tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(projected_tokens)
        if tokens.ndim == 2:
            tokens = tokens[None, ...]
        if self.config.input_format_type.lower() not in {"uint8", "auto"}:
            return tokens.astype(np.float32)
        if tokens.dtype == np.uint8:
            return tokens
        if self.config.token_uint8_mode == "dynamic_minmax":
            t = tokens.astype(np.float32)
            t_min = t.min(axis=(1, 2), keepdims=True)
            t_max = t.max(axis=(1, 2), keepdims=True)
            denom = np.maximum(t_max - t_min, 1e-6)
            q = np.clip(np.round((t - t_min) / denom * 255.0), 0, 255).astype(np.uint8)
            return q
        return np.clip(np.round(tokens), 0, 255).astype(np.uint8)

    def _build_inputs(
        self,
        current_image: np.ndarray,
        delayed_image: np.ndarray,
        projected_tokens: np.ndarray,
        goal_pose: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        tokens = self._prep_tokens(projected_tokens)
        inputs = {
            self.config.input_current_image: self._prep_image(current_image),
            self.config.input_delayed_image: self._prep_image(delayed_image),
            self.config.input_projected_tokens: tokens,
        }
        if self.config.input_goal_pose:
            goal = np.asarray(
                goal_pose if goal_pose is not None else np.zeros((3,), dtype=np.float32),
                dtype=np.float32,
            ).reshape(1, -1)
            inputs[self.config.input_goal_pose] = goal
        return inputs

    @staticmethod
    def _resolve_format_type(format_type_name: str, format_enum: Any) -> Any:
        name = format_type_name.strip().lower()
        if name == "float32":
            return format_enum.FLOAT32
        if name == "uint8":
            return format_enum.UINT8
        if name == "auto":
            return format_enum.AUTO
        raise ValueError(f"Unsupported format type: {format_type_name}")

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

        hef_path = str(Path(self.config.hef_path).expanduser().resolve())
        hef = HEF(hef_path)
        target = VDevice()
        self._target = target

        # Preferred path: VStreams. Some Hailo10H + HEF combinations may raise
        # HAILO_NOT_IMPLEMENTED on configure, so fallback to InferModel API.
        try:
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()
            input_format_type = self._resolve_format_type(self.config.input_format_type, FormatType)
            output_format_type = self._resolve_format_type(self.config.output_format_type, FormatType)
            input_params = InputVStreamParams.make(network_group, format_type=input_format_type)
            output_params = OutputVStreamParams.make(network_group, format_type=output_format_type)
            infer_pipeline = InferVStreams(network_group, input_params, output_params)

            self._network_group = network_group
            self._network_group_params = network_group_params
            self._infer_pipeline = infer_pipeline
            self._mode = "vstreams"
            self._ready = True
            return
        except Exception as exc:
            message = str(exc)
            if "HAILO_NOT_IMPLEMENTED" not in message:
                # Try InferModel fallback regardless; if it fails we re-raise later.
                pass

        infer_model = target.create_infer_model(hef_path)
        configured = infer_model.configure()
        self._infer_model = infer_model
        self._configured_infer_model = configured
        self._mode = "infer_model"
        self._ready = True

    @staticmethod
    def _align_rank_for_infer_model(array: np.ndarray, expected_rank: int) -> np.ndarray:
        if array.ndim == expected_rank:
            return array
        if array.ndim == expected_rank + 1 and array.shape[0] == 1:
            return array[0]
        return array

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
        if self._mode == "vstreams":
            with self._network_group.activate(self._network_group_params):
                output_dict = self._infer_pipeline.infer(inputs)
            if self.config.output_action_chunk in output_dict:
                output = output_dict[self.config.output_action_chunk]
            else:
                output = next(iter(output_dict.values()))
        else:
            configured = self._configured_infer_model
            bindings = configured.create_bindings()

            for name, value in inputs.items():
                expected_shape = tuple(self._infer_model.input(name).shape)
                arr = self._align_rank_for_infer_model(np.asarray(value), expected_rank=len(expected_shape))
                bindings.input(name).set_buffer(arr)

            output_name = self.config.output_action_chunk
            if output_name not in self._infer_model.output_names:
                output_name = self._infer_model.output_names[0]
            out_shape = tuple(self._infer_model.output(output_name).shape)
            out_dtype = np.uint8 if self.config.output_format_type.lower() in {"uint8", "auto"} else np.float32
            out_buf = np.empty(out_shape, dtype=out_dtype)
            bindings.output(output_name).set_buffer(out_buf)
            configured.run([bindings], 10000)
            output = out_buf

        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3 and output.shape[1] == 1 and output.shape[0] == self.config.chunk_size:
            output = output.reshape(1, self.config.chunk_size, -1)
        if output.ndim == 2:
            output = output[None, ...]
        if output.ndim != 3:
            output = output.reshape(1, self.config.chunk_size, self.config.pose_dim)
        return output
