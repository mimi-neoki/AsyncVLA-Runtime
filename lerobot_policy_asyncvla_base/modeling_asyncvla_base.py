from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from .configuration_asyncvla_base import AsyncVLABasePolicyConfig

try:
    from lerobot.common.policies.pretrained import PreTrainedPolicy
except Exception:  # pragma: no cover
    class PreTrainedPolicy(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()


ACTION_DIM = 4
NUM_ACTIONS_CHUNK = 8
ACTION_TOKENS_LEN = ACTION_DIM * NUM_ACTIONS_CHUNK
ACTION_TOKEN_BEGIN_IDX = 31743
IGNORE_INDEX = -100
STOP_TOKEN_ID = 2
TASK_MODE_TO_ID = {
    "satellite_only": 0,
    "pose_and_satellite": 1,
    "satellite_and_image": 2,
    "all": 3,
    "pose_only": 4,
    "pose_and_image": 5,
    "image_only": 6,
    "language_only": 7,
    "language_and_pose": 8,
}


@dataclass
class GuidancePacket:
    projected_tokens: np.ndarray
    timestamp_ns: int


class _OfficialActionTokenizer:
    def __init__(self, tokenizer: Any, bins: int = 256, min_action: int = -1, max_action: int = 1) -> None:
        self.tokenizer = tokenizer
        self.bins = np.linspace(min_action, max_action, bins)

    def __call__(self, action: np.ndarray) -> Any:
        action = np.clip(action, a_min=-1.0, a_max=1.0)
        discretized = np.digitize(action, self.bins)
        token_ids = self.tokenizer.vocab_size - discretized
        if len(discretized.shape) == 1:
            return self.tokenizer.decode(list(token_ids))
        return self.tokenizer.batch_decode(token_ids.tolist())


class _OfficialPurePromptBuilder:
    def __init__(self) -> None:
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> None:
        message = message.replace("<image>", "").strip()
        if (self.turn_count % 2) == 0:
            wrapped = f"In: {message}\nOut: "
        else:
            wrapped = f"{message if message != '' else ' '}</s>"
        self.prompt += wrapped
        self.turn_count += 1

    def get_prompt(self) -> str:
        return self.prompt.removeprefix("<s>").rstrip()


def _get_current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
    action_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(action_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    return mask * (token_ids > ACTION_TOKEN_BEGIN_IDX)


def _get_next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
    action_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(action_positions, dim=1)
    mask = cumsum > ACTION_DIM
    return mask * (token_ids > ACTION_TOKEN_BEGIN_IDX)


class ProprioProjector(nn.Module):
    """Project goal pose into the LLM embedding space."""

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(proprio)
        projected = self.act_fn1(projected)
        projected = self.fc2(projected)
        return projected


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNetIdCat(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, taskid: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = torch.cat((x, taskid.unsqueeze(1).unsqueeze(2).repeat(1, NUM_ACTIONS_CHUNK, 1)), dim=2)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class ProjActionTokens(nn.Module):
    """Token projector from base-VLA action hidden states to 1024-d guidance tokens."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 4096, action_dim: int = 1024) -> None:
        super().__init__()
        self.model = MLPResNetIdCat(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor, taskid: torch.Tensor) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        reshaped = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(reshaped, taskid)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def _load_torch_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return _strip_module_prefix(payload)


def _to_pil_image(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class AsyncVLABasePolicy(PreTrainedPolicy):
    """Workstation-side base-VLA policy that returns projected guidance tokens."""

    def __init__(self, config: AsyncVLABasePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        self.quantization_mode = self._resolve_quantization_mode(config.quantization)
        self.snapshot_dir = self.config.resolve_snapshot_dir()

        self.processor, self.base_model = self._load_base_runtime()
        self._validate_official_modality_contract()

        self.pose_projector = ProprioProjector(llm_dim=config.hidden_dim, proprio_dim=4).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.action_proj = ProjActionTokens(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.projected_dim,
        ).to(device=self.device, dtype=self.dtype)
        self._load_pose_projector_checkpoint()
        self._load_action_projector_checkpoint()
        self.pose_projector.eval()
        self.action_proj.eval()

        self.unnorm_key = self._resolve_unnorm_key()

    @classmethod
    def from_snapshot(cls, snapshot_dir: str, **kwargs: Any) -> "AsyncVLABasePolicy":
        cfg = AsyncVLABasePolicyConfig(snapshot_dir=snapshot_dir, **kwargs)
        return cls(cfg)

    def _resolve_device(self, requested: str | None) -> torch.device:
        normalized = str(requested or "auto").strip().lower()
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if has_mps:
                return torch.device("mps")
            return torch.device("cpu")

        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --device mps or --device cpu.")
        if device.type == "mps" and not has_mps:
            raise RuntimeError("MPS is not available. Use --device cuda or --device cpu.")
        return device

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        dtype = getattr(torch, dtype_name, torch.float16)
        if self.device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype

    def _resolve_quantization_mode(self, mode: str | None) -> str:
        normalized = str(mode or "none").strip().lower()
        aliases = {
            "none": "none",
            "off": "none",
            "no": "none",
            "false": "none",
            "0": "none",
            "8bit": "8bit",
            "int8": "8bit",
            "8": "8bit",
            "on": "8bit",
            "true": "8bit",
            "1": "8bit",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported quantization mode: {mode!r}. Expected one of: none, 8bit")
        return aliases[normalized]

    def _build_model_load_kwargs(self, transformers_module: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.quantization_mode == "none":
            kwargs["torch_dtype"] = self.dtype
            return kwargs

        if self.quantization_mode != "8bit":
            raise ValueError(f"Unsupported quantization mode: {self.quantization_mode}")
        if self.device.type != "cuda":
            raise RuntimeError("8bit quantization requires CUDA device. Set --device cuda or disable quantization.")

        bnb_cfg_cls = getattr(transformers_module, "BitsAndBytesConfig", None)
        if bnb_cfg_cls is None:
            raise RuntimeError("BitsAndBytesConfig is unavailable in transformers. Update transformers package.")
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("bitsandbytes is required for 8bit quantization. Install bitsandbytes.") from exc

        device_index = self.device.index if self.device.index is not None else 0
        kwargs["quantization_config"] = bnb_cfg_cls(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        # Work around transformers-openvla + accelerate behavior that calls `.to()`
        # on single-device maps for quantized models.
        kwargs["device_map"] = {"": device_index, "projector": "cpu"}
        return kwargs

    def _load_base_runtime(self) -> tuple[Any, nn.Module]:
        try:
            import transformers
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required to load AsyncVLA base model. "
                "Install dependencies from AsyncVLA first."
            ) from exc

        self._ensure_asyncvla_prismatic_namespaces()

        processor = AutoProcessor.from_pretrained(str(self.snapshot_dir), trust_remote_code=True)
        load_kwargs = self._build_model_load_kwargs(transformers)
        modeling_module = self._load_snapshot_package_module("modeling_prismatic")
        model_cls = getattr(modeling_module, "OpenVLAForActionPrediction_MMNv1", None)
        if model_cls is None:
            model = AutoModelForVision2Seq.from_pretrained(
                str(self.snapshot_dir),
                **load_kwargs,
            )
        else:
            model_load_kwargs = dict(load_kwargs)
            model_load_kwargs.pop("trust_remote_code", None)
            model = model_cls.from_pretrained(str(self.snapshot_dir), **model_load_kwargs)
        if hasattr(model, "vision_backbone") and hasattr(model.vision_backbone, "set_num_images_in_input"):
            model.vision_backbone.set_num_images_in_input(int(self.config.num_images_in_input))
        if self.quantization_mode == "none":
            model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        return processor, model

    def _register_namespace_package(self, package_name: str, package_dir: Path) -> None:
        if package_name in sys.modules:
            return
        module = types.ModuleType(package_name)
        module.__file__ = str(package_dir / "__init__.py")
        module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_dir)]
        module.__spec__ = spec
        sys.modules[package_name] = module

        if "." in package_name:
            parent_name, child_name = package_name.rsplit(".", 1)
            parent_mod = sys.modules.get(parent_name)
            if parent_mod is not None:
                setattr(parent_mod, child_name, module)

    def _ensure_asyncvla_prismatic_namespaces(self) -> None:
        repo_root = self.config.resolve_asyncvla_repo_dir()
        prismatic_root = repo_root / "prismatic"
        if not prismatic_root.exists():
            return

        # Avoid importing heavy package __init__ modules (which pull in training-time deps)
        # and expose only the namespace paths needed by HF remote code.
        self._register_namespace_package("prismatic", prismatic_root)
        training_root = prismatic_root / "training"
        if training_root.exists():
            self._register_namespace_package("prismatic.training", training_root)
        vla_root = prismatic_root / "vla"
        if vla_root.exists():
            self._register_namespace_package("prismatic.vla", vla_root)

    def _load_snapshot_package_module(self, module_stem: str) -> types.ModuleType:
        package_name = f"_asyncvla_snapshot_{abs(hash(str(self.snapshot_dir)))}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__file__ = str(self.snapshot_dir / "__init__.py")
            package.__path__ = [str(self.snapshot_dir)]  # type: ignore[attr-defined]
            spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(self.snapshot_dir)]
            package.__spec__ = spec
            sys.modules[package_name] = package

        full_name = f"{package_name}.{module_stem}"
        if full_name in sys.modules:
            return sys.modules[full_name]

        module_path = self.snapshot_dir / f"{module_stem}.py"
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load snapshot module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_pose_projector_checkpoint(self) -> None:
        path = self.config.resolve_pose_projector_path()
        if not path.exists():
            return
        state_dict = _load_torch_state_dict(path)
        self.pose_projector.load_state_dict(state_dict, strict=False)

    def _load_action_projector_checkpoint(self) -> None:
        path = self.config.resolve_projector_path()
        if not path.exists():
            return
        state_dict = _load_torch_state_dict(path)
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                normalized[key] = value
            elif key.startswith("action_proj.model."):
                normalized[key[len("action_proj.") :]] = value
        if normalized:
            self.action_proj.load_state_dict(normalized, strict=False)

    def _resolve_unnorm_key(self) -> str | None:
        if self.config.unnorm_key is not None:
            return self.config.unnorm_key
        norm_stats = getattr(self.base_model, "norm_stats", None)
        if isinstance(norm_stats, dict) and len(norm_stats) > 0:
            for key, dataset_stats in norm_stats.items():
                action_stats = dataset_stats.get("action", {}) if isinstance(dataset_stats, dict) else {}
                for candidate in ("q01", "min", "mean"):
                    values = action_stats.get(candidate)
                    if isinstance(values, (list, tuple)) and len(values) == ACTION_DIM:
                        return key

            # Fallback for AsyncVLA releases where bundled norm_stats are 7D but model action_dim is 4.
            runtime_key = "__asyncvla_runtime__"
            norm_stats[runtime_key] = {
                "action": {
                    "q01": [-1.0] * ACTION_DIM,
                    "q99": [1.0] * ACTION_DIM,
                    "min": [-1.0] * ACTION_DIM,
                    "max": [1.0] * ACTION_DIM,
                    "mask": [True] * ACTION_DIM,
                }
            }
            self.base_model.norm_stats = norm_stats
            return runtime_key
        return None

    def _validate_official_modality_contract(self) -> None:
        required_attrs = ["vision_backbone"]
        missing = [name for name in required_attrs if not hasattr(self.base_model, name)]
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                "Loaded base model does not support AsyncVLA official modality path. "
                f"Missing attributes: {missing_text}"
            )

    def _extract_image(self, observation: dict[str, Any], key: str) -> np.ndarray:
        value = observation.get(key)
        if value is None and isinstance(observation.get("images"), dict):
            value = observation["images"].get(key)
        if value is None:
            raise KeyError(f"Missing image key: {key}")
        return np.asarray(value)

    def _extract_goal_image(self, observation: dict[str, Any], current_image: np.ndarray) -> np.ndarray:
        goal_key = self.config.goal_image_key
        value = observation.get(goal_key)
        if value is None and isinstance(observation.get("images"), dict):
            value = observation["images"].get(goal_key)
        if value is None and self.config.duplicate_current_image_as_goal:
            return current_image
        if value is None:
            raise KeyError(
                f"Missing goal image key: {goal_key}. Set duplicate_current_image_as_goal=True or provide goal image."
            )
        return np.asarray(value)

    def _has_observation_image(self, observation: dict[str, Any], key: str) -> bool:
        if observation.get(key) is not None:
            return True
        images = observation.get("images")
        return isinstance(images, dict) and images.get(key) is not None

    def _task_mode_to_id(self, task_mode: Any) -> int:
        if isinstance(task_mode, (int, np.integer)):
            mode_id = int(task_mode)
            if 0 <= mode_id <= 8:
                return mode_id
            raise ValueError(f"task_mode id must be in [0, 8], got: {mode_id}")

        mode_raw = str(task_mode).strip().lower()
        aliases = {
            "0": "satellite_only",
            "1": "pose_and_satellite",
            "2": "satellite_and_image",
            "3": "all",
            "4": "pose_only",
            "5": "pose_and_image",
            "6": "image_only",
            "7": "language_only",
            "8": "language_and_pose",
            "satellite only": "satellite_only",
            "pose and satellite": "pose_and_satellite",
            "satellite and image": "satellite_and_image",
            "pose only": "pose_only",
            "pose and image": "pose_and_image",
            "image only": "image_only",
            "language only": "language_only",
            "language and pose": "language_and_pose",
            "satellite-only": "satellite_only",
            "pose-and-satellite": "pose_and_satellite",
            "satellite-and-image": "satellite_and_image",
            "pose-only": "pose_only",
            "pose-and-image": "pose_and_image",
            "image-only": "image_only",
            "language-only": "language_only",
            "language-and-pose": "language_and_pose",
        }
        canonical = aliases.get(mode_raw, mode_raw)
        if canonical not in TASK_MODE_TO_ID:
            valid = ", ".join(sorted(TASK_MODE_TO_ID.keys()))
            raise ValueError(f"Unsupported task_mode: {task_mode!r}. Expected one of: {valid} or ids 0-8.")
        return TASK_MODE_TO_ID[canonical]

    def _goal_pose_to_proprio(self, goal_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
        if pose.size >= 4:
            return pose[:4]
        if pose.size == 3:
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
            return np.array([x, y, np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        if pose.size == 2:
            x, y = float(pose[0]), float(pose[1])
            return np.array([x, y, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def _build_prompt(self, instruction: str, task_id: int) -> str:
        if task_id not in {TASK_MODE_TO_ID["language_only"], TASK_MODE_TO_ID["language_and_pose"]}:
            # Match official AsyncVLA non-language prompt behavior.
            return "No language instruction"
        task = instruction.strip()
        if not task:
            task = "move toward the goal"
        return f"What action should the robot take to {task}?"

    def _build_official_labels_batch(
        self,
        prompt_text: str,
        current_img_t: torch.Tensor,
        goal_img_t: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The official demo scaffolds a fake action response to recover the same label-based masks used in training.
        dummy_actions = np.random.default_rng(0).random((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
        action_tokenizer = _OfficialActionTokenizer(self.processor.tokenizer)
        current_action = dummy_actions[0]
        future_actions = dummy_actions[1:]
        future_actions_string = "".join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = _OfficialPurePromptBuilder()
        prompt_builder.add_turn("human", prompt_text)
        prompt_builder.add_turn("gpt", action_chunk_string)
        prompt = prompt_builder.get_prompt()

        tokenized = self.processor.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()
        labels[:, : -(action_chunk_len + 1)] = IGNORE_INDEX

        if goal_img_t is None:
            pixel_values = current_img_t.unsqueeze(0)
        else:
            pixel_values = torch.cat((current_img_t, goal_img_t), dim=0).unsqueeze(0)
        return input_ids, attention_mask, labels, pixel_values

    def _resolve_task_id(self, observation: dict[str, Any]) -> int:
        if self.config.task_id is not None:
            return int(self.config.task_id)
        request_task_id = observation.get(self.config.task_id_key)
        if request_task_id is not None:
            return self._task_mode_to_id(request_task_id)

        request_task_mode = observation.get(self.config.task_mode_key)
        if str(request_task_mode).strip().lower() not in {"none", "", "auto"}:
            return self._task_mode_to_id(request_task_mode)
        if str(self.config.task_mode).strip().lower() != "auto":
            return self._task_mode_to_id(self.config.task_mode)

        has_instruction = bool(str(observation.get(self.config.instruction_key, "")).strip())
        has_goal_pose = observation.get(self.config.goal_pose_key) is not None
        has_goal_image = self._has_observation_image(observation, self.config.goal_image_key)
        satellite = bool(observation.get(self.config.satellite_key, self.config.satellite_default))

        if satellite and not has_instruction and not has_goal_pose and not has_goal_image:
            return TASK_MODE_TO_ID["satellite_only"]
        if satellite and not has_instruction and has_goal_pose and not has_goal_image:
            return TASK_MODE_TO_ID["pose_and_satellite"]
        if satellite and not has_instruction and not has_goal_pose and has_goal_image:
            return TASK_MODE_TO_ID["satellite_and_image"]
        if satellite and not has_instruction and has_goal_pose and has_goal_image:
            return TASK_MODE_TO_ID["all"]
        if (not satellite) and (not has_instruction) and has_goal_pose and (not has_goal_image):
            return TASK_MODE_TO_ID["pose_only"]
        if (not satellite) and (not has_instruction) and has_goal_pose and has_goal_image:
            return TASK_MODE_TO_ID["pose_and_image"]
        if (not satellite) and (not has_instruction) and (not has_goal_pose) and has_goal_image:
            return TASK_MODE_TO_ID["image_only"]
        if (not satellite) and has_instruction and (not has_goal_pose) and (not has_goal_image):
            return TASK_MODE_TO_ID["language_only"]
        if (not satellite) and has_instruction and has_goal_pose and (not has_goal_image):
            return TASK_MODE_TO_ID["language_and_pose"]

        # Backward-compatible fallback for currently-supported modalities in this runtime.
        if has_instruction and has_goal_pose:
            return int(self.config.task_id_instruction_with_pose)
        if has_instruction:
            return int(self.config.task_id_instruction_only)
        if has_goal_pose:
            return int(self.config.task_id_pose_only)
        return int(self.config.task_id_image_only)

    def _prepare_model_inputs(self, observation: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        current_image_np = self._extract_image(observation, self.config.image_key)
        task_id = self._resolve_task_id(observation)

        prompt = self._build_prompt(str(observation.get(self.config.instruction_key, "")), task_id)

        current_img_t = self.processor.image_processor.apply_transform(_to_pil_image(current_image_np))
        if int(self.config.num_images_in_input) <= 1:
            goal_img_t = None
        else:
            goal_image_np = self._extract_goal_image(observation, current_image_np)
            goal_img_t = self.processor.image_processor.apply_transform(_to_pil_image(goal_image_np))
        input_ids, attention_mask, labels, pixel_values = self._build_official_labels_batch(
            prompt_text=prompt,
            current_img_t=current_img_t,
            goal_img_t=goal_img_t,
        )

        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
            "pixel_values": pixel_values.to(self.device, dtype=self.dtype),
            "task_id": torch.tensor([task_id], dtype=self.dtype, device=self.device),
        }
        return inputs, current_image_np

    def _predict_actions_hidden_states_official(
        self,
        model_inputs: dict[str, torch.Tensor],
        proprio: np.ndarray,
    ) -> torch.Tensor:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = model_inputs["labels"]
        pixel_values = model_inputs["pixel_values"]
        modality_id = model_inputs["task_id"]

        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            modality_id=modality_id,
            labels=labels,
            output_hidden_states=True,
            proprio=torch.as_tensor(proprio[None, :], device=self.device, dtype=self.dtype),
            proprio_projector=self.pose_projector,
            noisy_action_projector=None,
            use_film=False,
        )
        if getattr(output, "hidden_states", None) is None:
            raise RuntimeError("Base model forward did not return hidden_states")
        last_hidden_states = output.hidden_states[-1]
        ground_truth_token_ids = labels[:, 1:]
        current_action_mask = _get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = _get_next_actions_mask(ground_truth_token_ids)
        num_patches = int(self.base_model.vision_backbone.get_num_patches()) * int(
            self.base_model.vision_backbone.get_num_images_in_input()
        )
        num_patches += 1  # proprio token
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = int(input_ids.shape[0])
        return text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, ACTION_TOKENS_LEN, -1)

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> GuidancePacket:
        model_inputs, _ = self._prepare_model_inputs(observation)
        goal_pose_raw = np.asarray(observation.get(self.config.goal_pose_key, [0.0, 0.0, 0.0]), dtype=np.float32)
        proprio = self._goal_pose_to_proprio(goal_pose_raw)
        task_id_t = model_inputs["task_id"]
        actions_hidden_states = self._predict_actions_hidden_states_official(model_inputs, proprio)
        projected = self.action_proj.predict_action(actions_hidden_states, task_id_t)
        projected_np = projected.squeeze(0).detach().cpu().to(torch.float32).numpy()
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
    """Small helper to verify expected AsyncVLA snapshot members."""

    root = Path(snapshot_dir)
    expected = [
        "config.json",
        "model.safetensors.index.json",
        "modeling_prismatic.py",
        "processing_prismatic.py",
        "action_proj--750000_checkpoint.pt",
        "pose_projector--750000_checkpoint.pt",
    ]
    return {item: (root / item).exists() for item in expected}
