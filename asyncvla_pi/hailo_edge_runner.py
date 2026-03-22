from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .token_quant import load_token_quant_params, quantize_tokens_fixed_affine

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
    convert_bgr_to_rgb: bool = False
    # Historical name kept for compatibility; this now controls token quantization
    # for both uint8 and int8 input modes.
    token_uint8_mode: str = "dynamic_minmax"
    token_quant_params_path: str | None = None


class HailoEdgeRunner:
    """Runs edge adapter inference from HEF on the Pi side.

    If pyhailort is unavailable, you can pass `fallback_fn` for dry runs.
    """

    def __init__(
        self,
        config: HailoEdgeRunnerConfig,
        fallback_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
        target: Any | None = None,
    ) -> None:
        self.config = config
        self.fallback_fn = fallback_fn
        self._ready = False
        self._network_group: Any = None
        self._network_group_params: Any = None
        self._infer_pipeline: Any = None
        self._target: Any = target
        self._owns_target = target is None
        self._infer_model: Any = None
        self._configured_infer_model: Any = None
        self._quant_info_model: Any = None
        self._mode: str = "vstreams"
        self._resolved_input_current_name: str = config.input_current_image
        self._resolved_input_delayed_name: str = config.input_delayed_image
        self._resolved_input_tokens_name: str = config.input_projected_tokens
        self._resolved_input_goal_name: str | None = config.input_goal_pose
        self._resolved_output_name: str = config.output_action_chunk
        self._token_quant_params: dict[str, np.ndarray | float | int] | None = None
        if config.token_quant_params_path:
            self._token_quant_params = load_token_quant_params(config.token_quant_params_path)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(image, (self.config.image_width, self.config.image_height))
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, self.config.image_height).astype(np.int32)
        x_idx = np.linspace(0, w - 1, self.config.image_width).astype(np.int32)
        return image[np.ix_(y_idx, x_idx)]

    def _input_format_name(self) -> str:
        return self.config.input_format_type.strip().lower()

    @staticmethod
    def _quantize_signed_dynamic_minmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        v_min = arr.min(axis=tuple(range(1, arr.ndim)), keepdims=True)
        v_max = arr.max(axis=tuple(range(1, arr.ndim)), keepdims=True)
        denom = np.maximum(v_max - v_min, 1e-6)
        scaled = (arr - v_min) / denom * 255.0 - 128.0
        return np.clip(np.round(scaled), -128, 127).astype(np.int8)

    @staticmethod
    def _quantize_unsigned_dynamic_minmax(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        v_min = arr.min(axis=tuple(range(1, arr.ndim)), keepdims=True)
        v_max = arr.max(axis=tuple(range(1, arr.ndim)), keepdims=True)
        denom = np.maximum(v_max - v_min, 1e-6)
        scaled = (arr - v_min) / denom * 255.0
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)

    def _prep_image(self, image: np.ndarray) -> np.ndarray:
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
        resized = self._resize(arr)
        format_name = self._input_format_name()
        if format_name in {"uint8", "auto"}:
            data = resized.astype(np.uint8)
        elif format_name == "int8":
            data = np.clip(resized.astype(np.int16) - 128, -128, 127).astype(np.int8)
        else:
            data = resized.astype(np.float32)
        if self.config.image_scale_255 and not np.issubdtype(data.dtype, np.integer) and data.max() > 1.0:
            data = data / 255.0
        if self.config.normalize_imagenet and not np.issubdtype(data.dtype, np.integer):
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
        format_name = self._input_format_name()
        if format_name not in {"uint8", "int8", "auto"}:
            return tokens.astype(np.float32)
        if format_name in {"uint8", "auto"} and tokens.dtype == np.uint8:
            return tokens
        if format_name == "int8" and tokens.dtype == np.int8:
            return tokens
        token_mode = self.config.token_uint8_mode.strip().lower()
        if token_mode == "dynamic_minmax":
            if format_name == "int8":
                return self._quantize_signed_dynamic_minmax(tokens)
            return self._quantize_unsigned_dynamic_minmax(tokens)
        if token_mode == "fixed_affine":
            if self._token_quant_params is None:
                raise ValueError("token_uint8_mode='fixed_affine' requires token_quant_params_path")
            return quantize_tokens_fixed_affine(
                tokens,
                quant_dtype="int8" if format_name == "int8" else "uint8",
                scales=np.asarray(self._token_quant_params["scales"], dtype=np.float32),
                zero_point=int(self._token_quant_params["zero_point"]),
            )
        if format_name == "int8":
            return np.clip(np.round(tokens), -128, 127).astype(np.int8)
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
        if name == "int8":
            return format_enum.INT8
        if name == "uint8":
            return format_enum.UINT8
        if name == "auto":
            return format_enum.AUTO
        raise ValueError(f"Unsupported format type: {format_type_name}")

    @staticmethod
    def _resolve_stream_name(requested_name: str | None, available_names: list[str]) -> str | None:
        if requested_name is None:
            return None
        if not available_names:
            return requested_name
        if requested_name in available_names:
            return requested_name
        requested_leaf = requested_name.split("/")[-1]
        leaf_matches = [name for name in available_names if name.split("/")[-1] == requested_leaf]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        if len(leaf_matches) > 1:
            # deterministic choice for ambiguous matches
            return sorted(leaf_matches)[0]
        return requested_name

    def _resolve_stream_names(self, available_inputs: list[str], available_outputs: list[str]) -> None:
        self._resolved_input_current_name = self._resolve_stream_name(
            self.config.input_current_image, available_inputs
        ) or self.config.input_current_image
        self._resolved_input_delayed_name = self._resolve_stream_name(
            self.config.input_delayed_image, available_inputs
        ) or self.config.input_delayed_image
        self._resolved_input_tokens_name = self._resolve_stream_name(
            self.config.input_projected_tokens, available_inputs
        ) or self.config.input_projected_tokens
        self._resolved_input_goal_name = self._resolve_stream_name(
            self.config.input_goal_pose, available_inputs
        )
        self._resolved_output_name = self._resolve_stream_name(
            self.config.output_action_chunk, available_outputs
        ) or self.config.output_action_chunk

    def _map_input_name(self, logical_name: str) -> str:
        if logical_name == self.config.input_current_image:
            return self._resolved_input_current_name
        if logical_name == self.config.input_delayed_image:
            return self._resolved_input_delayed_name
        if logical_name == self.config.input_projected_tokens:
            return self._resolved_input_tokens_name
        if self.config.input_goal_pose and logical_name == self.config.input_goal_pose:
            return self._resolved_input_goal_name or logical_name
        return logical_name

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
        target = self._target
        if target is None:
            target = VDevice()
            self._target = target
            self._owns_target = True
        # Keep an InferModel handle for quant metadata even when VStreams is used.
        try:
            self._quant_info_model = target.create_infer_model(hef_path)
        except Exception:
            self._quant_info_model = None

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
            available_inputs = list(input_params.keys()) if hasattr(input_params, "keys") else []
            available_outputs = list(output_params.keys()) if hasattr(output_params, "keys") else []
            self._resolve_stream_names(available_inputs, available_outputs)
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

        infer_model = self._quant_info_model
        if infer_model is None:
            infer_model = target.create_infer_model(hef_path)
            self._quant_info_model = infer_model
        configured = infer_model.configure()
        self._infer_model = infer_model
        self._configured_infer_model = configured
        self._resolve_stream_names(list(infer_model.input_names), list(infer_model.output_names))
        self._mode = "infer_model"
        self._ready = True

    def _get_output_quant_info(self, output_name: str) -> tuple[float, float] | None:
        model = self._infer_model if self._infer_model is not None else self._quant_info_model
        if model is None:
            return None
        candidate_names = [output_name, self._resolved_output_name, self.config.output_action_chunk]
        seen: set[str] = set()
        for name in candidate_names:
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                quant_infos = model.output(name).quant_infos
            except Exception:
                continue
            if quant_infos:
                qinfo = quant_infos[0]
                return float(qinfo.qp_scale), float(qinfo.qp_zp)
        return None

    def _dequantize_output(self, output_name: str, output: np.ndarray) -> np.ndarray:
        """Dequantize integer output buffers to float32 using Hailo quant metadata."""
        if not np.issubdtype(np.asarray(output).dtype, np.integer):
            return np.asarray(output, dtype=np.float32)
        qinfo = self._get_output_quant_info(output_name)
        if qinfo is None:
            return np.asarray(output, dtype=np.float32)
        scale, zp = qinfo
        return (np.asarray(output, dtype=np.float32) - zp) * scale

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
            mapped_inputs = {self._map_input_name(name): value for name, value in inputs.items()}
            with self._network_group.activate(self._network_group_params):
                output_dict = self._infer_pipeline.infer(mapped_inputs)
            if self._resolved_output_name in output_dict:
                output = output_dict[self._resolved_output_name]
            elif self.config.output_action_chunk in output_dict:
                output = output_dict[self.config.output_action_chunk]
            else:
                output = next(iter(output_dict.values()))
        else:
            configured = self._configured_infer_model
            bindings = configured.create_bindings()

            for name, value in inputs.items():
                resolved_name = self._map_input_name(name)
                expected_shape = tuple(self._infer_model.input(resolved_name).shape)
                arr = self._align_rank_for_infer_model(np.asarray(value), expected_rank=len(expected_shape))
                bindings.input(resolved_name).set_buffer(arr)

            output_name = self._resolved_output_name
            if output_name not in self._infer_model.output_names and self.config.output_action_chunk in self._infer_model.output_names:
                output_name = self.config.output_action_chunk
            if output_name not in self._infer_model.output_names:
                output_name = self._infer_model.output_names[0]
            out_shape = tuple(self._infer_model.output(output_name).shape)
            # InferModel expects a native-quantized buffer shape/size.
            # Use a quantized integer buffer and dequantize manually when float output is requested.
            if self.config.output_format_type.lower() == "float32":
                out_dtype = np.uint8
            else:
                if self.config.output_format_type.lower() == "int8":
                    out_dtype = np.int8
                elif self.config.output_format_type.lower() in {"uint8", "auto"}:
                    out_dtype = np.uint8
                else:
                    out_dtype = np.float32
            out_buf = np.empty(out_shape, dtype=out_dtype)
            bindings.output(output_name).set_buffer(out_buf)
            configured.run([bindings], 10000)
            output = out_buf

        if self._mode == "infer_model":
            if self.config.output_format_type.lower() == "float32":
                output = self._dequantize_output(output_name, output)
            else:
                output = np.asarray(output, dtype=np.float32)
        else:
            if self.config.output_format_type.lower() == "float32":
                output = self._dequantize_output(self._resolved_output_name, output)
            else:
                output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3 and output.shape[1] == 1 and output.shape[0] == self.config.chunk_size:
            output = output.reshape(1, self.config.chunk_size, -1)
        if output.ndim == 2:
            output = output[None, ...]
        if output.ndim != 3:
            output = output.reshape(1, self.config.chunk_size, self.config.pose_dim)
        return output

    def close(self) -> None:
        self._network_group = None
        self._network_group_params = None
        self._infer_pipeline = None
        self._infer_model = None
        self._configured_infer_model = None
        self._quant_info_model = None
        self._ready = False
        if self._owns_target:
            self._target = None
