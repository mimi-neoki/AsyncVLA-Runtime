from __future__ import annotations

import sys
import types

import numpy as np
from PIL import Image
import torch

from lerobot_policy_asyncvla_base.configuration_asyncvla_base import AsyncVLABasePolicyConfig
from lerobot_policy_asyncvla_base.modeling_asyncvla_base import (
    ACTION_TOKEN_BEGIN_IDX,
    AsyncVLABasePolicy,
    ProjActionTokens,
    ProprioProjector,
)
from scripts.run_base_vla_server import _prepare_observation


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_text = ""
        self.vocab_size = 32000
        self.pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        self.last_text = text
        ids_list: list[int] = [1]
        if "Out: " in text:
            prefix, suffix = text.split("Out: ", 1)
            for ch in prefix:
                ids_list.append(100 + (ord(ch) % 50))
            response, _, tail = suffix.partition("</s>")
            for idx, _ in enumerate(response):
                ids_list.append(ACTION_TOKEN_BEGIN_IDX + 1 + (idx % 4))
            if tail:
                for ch in tail:
                    ids_list.append(200 + (ord(ch) % 50))
            ids_list.append(2)
        else:
            for ch in text:
                ids_list.append(100 + (ord(ch) % 50))
        ids = torch.tensor([ids_list], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids) -> str:
        return "".join(chr(65 + (int(tok) % 26)) for tok in token_ids)

    def batch_decode(self, batch_token_ids) -> list[str]:
        return [self.decode(token_ids) for token_ids in batch_token_ids]


class _FakeImageProcessor:
    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        return torch.ones(3, h, w, dtype=torch.float32)


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()


class _FakeVisionBackbone:
    def __init__(self) -> None:
        self.num_images_in_input = 1
        self.num_patches = 4

    def set_num_images_in_input(self, value: int) -> None:
        self.num_images_in_input = value

    def get_num_images_in_input(self) -> int:
        return int(self.num_images_in_input)

    def get_num_patches(self) -> int:
        return int(self.num_patches)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_stats = {"dummy_dataset": {}}
        self.vision_backbone = _FakeVisionBackbone()
        self.last_forward_kwargs: dict[str, object] | None = None

    def eval(self) -> "_FakeModel":
        return self

    def to(self, *args, **kwargs) -> "_FakeModel":
        return self

    def forward(self, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size = int(input_ids.shape[0])
        labels = kwargs["labels"]
        seq_len = int(input_ids.shape[1])
        num_patches = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input() + 1
        hidden = torch.zeros(batch_size, seq_len + num_patches, 16, dtype=torch.float32, device=input_ids.device)
        active = labels != -100
        for idx in range(batch_size):
            active_positions = torch.where(active[idx])[0]
            for offset, pos in enumerate(active_positions.tolist()):
                hidden[idx, num_patches + pos, :] = float(offset + 1)
        return types.SimpleNamespace(hidden_states=(hidden,))


def test_prepare_observation_decodes_all_images() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    payload = {
        "timestamp_ns": 123,
        "task_mode": "language_and_pose",
        "task_id": 8,
        "satellite": False,
        "images": {
            "front_image": {"data": img.tolist(), "shape": list(img.shape)},
            "goal_image": {"data": img.tolist(), "shape": list(img.shape)},
        },
    }
    obs = _prepare_observation(payload, image_key="front_image")
    assert obs["timestamp_ns"] == 123
    assert "front_image" in obs["images"]
    assert "goal_image" in obs["images"]
    assert obs["images"]["front_image"].shape == (8, 8, 3)
    assert obs["task_mode"] == "language_and_pose"
    assert obs["task_id"] == 8
    assert obs["satellite"] is False


def test_base_policy_infer_with_mock_runtime(monkeypatch, tmp_path) -> None:
    config = AsyncVLABasePolicyConfig(
        snapshot_dir=str(tmp_path),
        device="cpu",
        dtype="float32",
        hidden_dim=16,
        projected_dim=6,
        num_images_in_input=2,
    )

    # Create projector checkpoints with the same naming as AsyncVLA_release.
    action_proj = ProjActionTokens(input_dim=16, hidden_dim=16, action_dim=6)
    torch.save({f"module.{k}": v for k, v in action_proj.state_dict().items()}, tmp_path / config.projector_checkpoint)
    pose_proj = ProprioProjector(llm_dim=16, proprio_dim=4)
    torch.save({f"module.{k}": v for k, v in pose_proj.state_dict().items()}, tmp_path / config.pose_projector_checkpoint)

    def _fake_load_base_runtime(self):
        processor = _FakeProcessor()
        model = _FakeModel()
        model.vision_backbone.set_num_images_in_input(self.config.num_images_in_input)
        return processor, model

    monkeypatch.setattr(AsyncVLABasePolicy, "_load_base_runtime", _fake_load_base_runtime)
    policy = AsyncVLABasePolicy(config)

    observation = {
        "front_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "goal_pose": [1.0, 2.0, 0.5],
        "instruction": "move toward blue bin",
        "timestamp_ns": 987654321,
    }
    result = policy.select_action(observation)

    assert result["timestamp_ns"] == 987654321
    assert result["projected_tokens"].shape == (8, 6)
    assert np.isfinite(result["projected_tokens"]).all()
    assert "What action should the robot take to move toward blue bin?" in policy.processor.tokenizer.last_text


def test_base_policy_uses_official_modality_forward_path(monkeypatch, tmp_path) -> None:
    config = AsyncVLABasePolicyConfig(
        snapshot_dir=str(tmp_path),
        device="cpu",
        dtype="float32",
        hidden_dim=16,
        projected_dim=6,
        num_images_in_input=2,
    )

    action_proj = ProjActionTokens(input_dim=16, hidden_dim=16, action_dim=6)
    torch.save({f"module.{k}": v for k, v in action_proj.state_dict().items()}, tmp_path / config.projector_checkpoint)
    pose_proj = ProprioProjector(llm_dim=16, proprio_dim=4)
    torch.save({f"module.{k}": v for k, v in pose_proj.state_dict().items()}, tmp_path / config.pose_projector_checkpoint)

    fake_model = _FakeModel()

    def _fake_load_base_runtime(self):
        processor = _FakeProcessor()
        fake_model.vision_backbone.set_num_images_in_input(self.config.num_images_in_input)
        return processor, fake_model

    monkeypatch.setattr(AsyncVLABasePolicy, "_load_base_runtime", _fake_load_base_runtime)
    policy = AsyncVLABasePolicy(config)

    observation = {
        "front_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "goal_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "task_mode": "image_only",
        "timestamp_ns": 42,
    }
    result = policy.select_action(observation)

    assert result["timestamp_ns"] == 42
    assert result["projected_tokens"].shape == (8, 6)
    assert "No language instruction" in policy.processor.tokenizer.last_text
    assert fake_model.last_forward_kwargs is not None
    assert "modality_id" in fake_model.last_forward_kwargs
    modality_id = fake_model.last_forward_kwargs["modality_id"]
    assert isinstance(modality_id, torch.Tensor)
    assert int(modality_id.item()) == 6
    assert "labels" in fake_model.last_forward_kwargs


def test_prismatic_namespace_bootstrap(monkeypatch, tmp_path) -> None:
    repo_dir = tmp_path / "AsyncVLA"
    (repo_dir / "prismatic" / "training").mkdir(parents=True)
    (repo_dir / "prismatic" / "vla").mkdir(parents=True)

    config = AsyncVLABasePolicyConfig(
        snapshot_dir=str(tmp_path),
        asyncvla_repo_dir=str(repo_dir),
    )
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = config

    monkeypatch.delitem(sys.modules, "prismatic", raising=False)
    monkeypatch.delitem(sys.modules, "prismatic.training", raising=False)
    monkeypatch.delitem(sys.modules, "prismatic.vla", raising=False)

    policy._ensure_asyncvla_prismatic_namespaces()

    assert "prismatic" in sys.modules
    assert "prismatic.training" in sys.modules
    assert "prismatic.vla" in sys.modules


def test_quantization_none_uses_torch_dtype(tmp_path) -> None:
    config = AsyncVLABasePolicyConfig(snapshot_dir=str(tmp_path), quantization="none", dtype="float16")
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = config
    policy.device = torch.device("cuda:0")
    policy.dtype = torch.float16
    policy.quantization_mode = policy._resolve_quantization_mode(config.quantization)

    fake_transformers = types.SimpleNamespace(BitsAndBytesConfig=None)
    kwargs = policy._build_model_load_kwargs(fake_transformers)
    assert kwargs["torch_dtype"] == torch.float16
    assert "quantization_config" not in kwargs


def test_quantization_8bit_builds_bnb_kwargs(monkeypatch, tmp_path) -> None:
    config = AsyncVLABasePolicyConfig(snapshot_dir=str(tmp_path), quantization="8bit", dtype="float16")
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = config
    policy.device = torch.device("cuda:0")
    policy.dtype = torch.float16
    policy.quantization_mode = policy._resolve_quantization_mode(config.quantization)

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.SimpleNamespace(BitsAndBytesConfig=_FakeBitsAndBytesConfig)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))

    kwargs = policy._build_model_load_kwargs(fake_transformers)
    assert kwargs["device_map"] == {"": 0, "projector": "cpu"}
    assert kwargs["quantization_config"].kwargs["load_in_8bit"] is True
    assert kwargs["quantization_config"].kwargs["llm_int8_enable_fp32_cpu_offload"] is True
    assert "torch_dtype" not in kwargs


def test_resolve_unnorm_key_injects_runtime_stats_for_action_dim(tmp_path) -> None:
    config = AsyncVLABasePolicyConfig(snapshot_dir=str(tmp_path))
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = config
    policy.base_model = types.SimpleNamespace(
        norm_stats={
            "bridge_orig": {
                "action": {
                    "q01": [0.0] * 7,
                    "q99": [1.0] * 7,
                }
            }
        }
    )

    key = policy._resolve_unnorm_key()
    assert key == "__asyncvla_runtime__"
    runtime_stats = policy.base_model.norm_stats[key]["action"]
    assert len(runtime_stats["q01"]) == 4


def test_task_mode_to_id_accepts_run_asyncvla_labels(tmp_path) -> None:
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = AsyncVLABasePolicyConfig(snapshot_dir=str(tmp_path))
    assert policy._task_mode_to_id("satellite only") == 0
    assert policy._task_mode_to_id("pose_and_satellite") == 1
    assert policy._task_mode_to_id("satellite-and-image") == 2
    assert policy._task_mode_to_id("all") == 3
    assert policy._task_mode_to_id("pose_only") == 4
    assert policy._task_mode_to_id("pose and image") == 5
    assert policy._task_mode_to_id("image_only") == 6
    assert policy._task_mode_to_id("language only") == 7
    assert policy._task_mode_to_id("language_and_pose") == 8


def test_resolve_task_id_prefers_request_task_mode(tmp_path) -> None:
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = AsyncVLABasePolicyConfig(
        snapshot_dir=str(tmp_path),
        task_mode="pose_only",
        task_id=None,
    )
    observation = {"task_mode": "language_only"}
    assert policy._resolve_task_id(observation) == 7


def test_resolve_task_id_auto_mapping_matches_run_asyncvla(tmp_path) -> None:
    policy = AsyncVLABasePolicy.__new__(AsyncVLABasePolicy)
    policy.config = AsyncVLABasePolicyConfig(snapshot_dir=str(tmp_path), task_mode="auto", task_id=None)

    assert policy._resolve_task_id({"satellite": True}) == 0
    assert policy._resolve_task_id({"satellite": True, "goal_pose": [1, 0, 0]}) == 1
    assert policy._resolve_task_id({"satellite": True, "images": {"goal_image": np.zeros((2, 2, 3), dtype=np.uint8)}}) == 2
    assert policy._resolve_task_id(
        {"satellite": True, "goal_pose": [1, 0, 0], "images": {"goal_image": np.zeros((2, 2, 3), dtype=np.uint8)}}
    ) == 3
    assert policy._resolve_task_id({"goal_pose": [1, 0, 0]}) == 4
    assert policy._resolve_task_id({"goal_pose": [1, 0, 0], "images": {"goal_image": np.zeros((2, 2, 3), dtype=np.uint8)}}) == 5
    assert policy._resolve_task_id({"images": {"goal_image": np.zeros((2, 2, 3), dtype=np.uint8)}}) == 6
    assert policy._resolve_task_id({"instruction": "go"}) == 7
    assert policy._resolve_task_id({"instruction": "go", "goal_pose": [1, 0, 0]}) == 8
