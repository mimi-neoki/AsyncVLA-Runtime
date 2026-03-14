#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asyncvla_pi.edge_adapter_model import load_edge_adapter_from_hf_snapshot
from lerobot_policy_asyncvla_base.modeling_asyncvla_base import AsyncVLABasePolicy

ACTION_DIM = 4
NUM_ACTIONS_CHUNK = 8
ACTION_TOKEN_BEGIN_IDX = 31743
IGNORE_INDEX = -100
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare official AsyncVLA demo path and runtime path outputs")
    parser.add_argument("--hf-dir", default="~/huggingface/AsyncVLA_release")
    parser.add_argument("--asyncvla-repo-dir", default="~/gitrepo/AsyncVLA")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--metric-waypoint-spacing", type=float, default=0.1)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=-10.0)
    parser.add_argument("--goal-yaw-deg", type=float, default=-90.0)
    parser.add_argument("--instruction", default="move toward blue trash bin")
    parser.add_argument("--modes", nargs="*", choices=sorted(TASK_MODE_TO_ID.keys()), default=sorted(TASK_MODE_TO_ID.keys()))
    parser.add_argument("--save-json", default="artifacts/compare_official_demo_outputs.json")
    return parser.parse_args()


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


def _load_demo_images(asyncvla_repo_dir: Path) -> tuple[Image.Image, Image.Image, Image.Image]:
    inference_dir = asyncvla_repo_dir / "inference"
    past = Image.open(inference_dir / "past.png").convert("RGB").resize((224, 224), Image.BILINEAR)
    goal = Image.open(inference_dir / "goal.png").convert("RGB").resize((224, 224), Image.BILINEAR)
    cur = Image.open(inference_dir / "cur.png").convert("RGB").resize((224, 224), Image.BILINEAR)
    return past, goal, cur


def _goal_pose_loc_norm(goal_x: float, goal_y: float, goal_yaw_deg: float, spacing: float) -> np.ndarray:
    yaw_rad = math.radians(goal_yaw_deg)
    return np.asarray(
        [goal_x / spacing, goal_y / spacing, math.cos(yaw_rad), math.sin(yaw_rad)],
        dtype=np.float32,
    )


def _build_observation(
    past: Image.Image,
    goal: Image.Image,
    goal_pose_loc_norm: np.ndarray,
    task_mode: str,
    instruction: str,
) -> dict[str, Any]:
    observation = {
        "images": {
            "front_image": np.asarray(past, dtype=np.uint8),
            "goal_image": np.asarray(goal, dtype=np.uint8),
        },
        "goal_pose": goal_pose_loc_norm.tolist(),
        "task_mode": task_mode,
    }
    if task_mode in {"language_only", "language_and_pose"}:
        observation["instruction"] = instruction
    if task_mode in {"satellite_only", "pose_and_satellite", "satellite_and_image", "all"}:
        observation["satellite"] = True
    return observation


def _official_build_batch(
    tokenizer: Any,
    processor: Any,
    current_image_pil: Image.Image,
    goal_image_pil: Image.Image,
    goal_pose_loc_norm: np.ndarray,
    prompt_text: str,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    actions = rng.random((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
    action_tokenizer = _OfficialActionTokenizer(tokenizer)
    current_action = actions[0]
    future_actions = actions[1:]
    future_actions_string = "".join(action_tokenizer(future_actions))
    current_action_string = action_tokenizer(current_action)
    action_chunk_string = current_action_string + future_actions_string
    action_chunk_len = len(action_chunk_string)

    prompt_builder = _OfficialPurePromptBuilder()
    prompt_builder.add_turn("human", prompt_text)
    prompt_builder.add_turn("gpt", action_chunk_string)

    input_ids = torch.tensor(tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
    labels = input_ids.clone()
    labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
    pixel_values_current = processor.image_processor.apply_transform(current_image_pil)
    pixel_values_goal = processor.image_processor.apply_transform(goal_image_pil)

    batch = {
        "input_ids": pad_sequence([input_ids], batch_first=True, padding_value=tokenizer.pad_token_id),
        "labels": pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX),
        "pixel_values": torch.cat((pixel_values_current.unsqueeze(0), pixel_values_goal.unsqueeze(0)), dim=1),
        "goal_pose": torch.stack([torch.from_numpy(goal_pose_loc_norm.copy())]),
    }
    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.pad_token_id)
    return batch


def _tensor_diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a32 = a.detach().cpu().to(torch.float32)
    b32 = b.detach().cpu().to(torch.float32)
    diff = a32 - b32
    flat_a = a32.reshape(-1)
    flat_b = b32.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
    return {
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "cosine_similarity": float(cosine),
    }


def _edge_preprocess(image: Image.Image) -> torch.Tensor:
    resized = image.resize((96, 96), Image.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]
    poses = [torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)]
    for t in range(1, dx.shape[1]):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)


def _predict_edge_trajectories(
    hf_dir: Path,
    device: torch.device,
    projected_tokens: torch.Tensor,
    past: Image.Image,
    cur: Image.Image,
    metric_waypoint_spacing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    shead, _, _, _ = load_edge_adapter_from_hf_snapshot(hf_dir)
    shead = shead.to(device=device, dtype=torch.float32).eval()
    projected_tokens = projected_tokens.to(device=device, dtype=torch.float32)
    if projected_tokens.ndim == 2:
        projected_tokens = projected_tokens.unsqueeze(0)

    img_past = _edge_preprocess(past).to(device)
    img_cur = _edge_preprocess(cur).to(device)
    with torch.no_grad():
        pose_past = _delta_to_pose(shead(img_past, img_past, projected_tokens)).cpu().to(torch.float32)
        pose_cur = _delta_to_pose(shead(img_cur, img_past, projected_tokens)).cpu().to(torch.float32)
    pose_past[..., :2] *= float(metric_waypoint_spacing)
    pose_cur[..., :2] *= float(metric_waypoint_spacing)
    return pose_past, pose_cur


def _official_prompt_for_mode(task_mode: str, instruction: str) -> str:
    if task_mode in {"language_only", "language_and_pose"}:
        return f"What action should the robot take to {instruction}?"
    return "No language instruction"


def main() -> int:
    args = parse_args()
    hf_dir = Path(args.hf_dir).expanduser().resolve()
    asyncvla_repo_dir = Path(args.asyncvla_repo_dir).expanduser().resolve()

    device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    past_img, goal_img, cur_img = _load_demo_images(asyncvla_repo_dir)
    goal_pose_loc_norm = _goal_pose_loc_norm(
        goal_x=float(args.goal_x),
        goal_y=float(args.goal_y),
        goal_yaw_deg=float(args.goal_yaw_deg),
        spacing=float(args.metric_waypoint_spacing),
    )
    modes = [str(mode) for mode in args.modes]

    policy = AsyncVLABasePolicy.from_snapshot(
        str(hf_dir),
        asyncvla_repo_dir=str(asyncvla_repo_dir),
        device=device.type,
        dtype=args.dtype,
        num_images_in_input=2,
        task_mode="auto",
        duplicate_current_image_as_goal=False,
    )
    policy.eval()

    per_mode: dict[str, Any] = {}
    for task_mode in modes:
        observation = _build_observation(
            past=past_img,
            goal=goal_img,
            goal_pose_loc_norm=goal_pose_loc_norm,
            task_mode=task_mode,
            instruction=str(args.instruction),
        )
        model_inputs, _ = policy._prepare_model_inputs(observation)
        proprio = policy._goal_pose_to_proprio(np.asarray(observation["goal_pose"], dtype=np.float32))
        runtime_hidden = policy._predict_actions_hidden_states_official(model_inputs, proprio)
        runtime_projected = policy.action_proj.predict_action(runtime_hidden, model_inputs["task_id"])

        official_batch = _official_build_batch(
            tokenizer=policy.processor.tokenizer,
            processor=policy.processor,
            current_image_pil=past_img,
            goal_image_pil=goal_img,
            goal_pose_loc_norm=goal_pose_loc_norm,
            prompt_text=_official_prompt_for_mode(task_mode, str(args.instruction)),
        )
        modality_id = torch.tensor([float(TASK_MODE_TO_ID[task_mode])], dtype=policy.dtype, device=policy.device)

        with torch.no_grad():
            official_output = policy.base_model(
                input_ids=official_batch["input_ids"].to(policy.device),
                attention_mask=official_batch["attention_mask"].to(policy.device),
                pixel_values=official_batch["pixel_values"].to(policy.device, dtype=policy.dtype),
                modality_id=modality_id,
                labels=official_batch["labels"].to(policy.device),
                output_hidden_states=True,
                proprio=official_batch["goal_pose"].to(policy.device, dtype=policy.dtype),
                proprio_projector=policy.pose_projector,
                noisy_action_projector=None,
                use_film=False,
            )

        ground_truth_token_ids = official_batch["labels"][:, 1:].to(policy.device)
        current_action_mask = _get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = _get_next_actions_mask(ground_truth_token_ids)
        last_hidden_states = official_output.hidden_states[-1]
        num_patches = int(policy.base_model.vision_backbone.get_num_patches()) * int(
            policy.base_model.vision_backbone.get_num_images_in_input()
        )
        num_patches += 1
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        official_hidden = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(official_batch["input_ids"].shape[0], 32, -1)
            .to(policy.dtype)
        )
        official_projected = policy.action_proj.predict_action(official_hidden, modality_id)

        runtime_edge_past, runtime_edge_cur = _predict_edge_trajectories(
            hf_dir=hf_dir,
            device=device,
            projected_tokens=runtime_projected,
            past=past_img,
            cur=cur_img,
            metric_waypoint_spacing=float(args.metric_waypoint_spacing),
        )
        official_edge_past, official_edge_cur = _predict_edge_trajectories(
            hf_dir=hf_dir,
            device=device,
            projected_tokens=official_projected,
            past=past_img,
            cur=cur_img,
            metric_waypoint_spacing=float(args.metric_waypoint_spacing),
        )

        per_mode[task_mode] = {
            "modality_id": TASK_MODE_TO_ID[task_mode],
            "runtime_vs_official": {
                "actions_hidden_states": _tensor_diff_stats(runtime_hidden, official_hidden),
                "projected_tokens": _tensor_diff_stats(runtime_projected, official_projected),
                "edge_pose_past": _tensor_diff_stats(runtime_edge_past, official_edge_past),
                "edge_pose_cur": _tensor_diff_stats(runtime_edge_cur, official_edge_cur),
            },
        }

    report = {
        "device": device.type,
        "dtype": args.dtype,
        "goal_pose_loc_norm": goal_pose_loc_norm.tolist(),
        "modes": per_mode,
    }

    save_path = Path(args.save_json).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"device: {report['device']} dtype: {report['dtype']}")
    print(f"goal_pose_loc_norm: {report['goal_pose_loc_norm']}")
    for task_mode in modes:
        mode_report = report["modes"][task_mode]
        print(f"mode: {task_mode} modality_id: {mode_report['modality_id']}")
        for key, stats in mode_report["runtime_vs_official"].items():
            print(
                f"  {key}: "
                f"mean_abs={stats['mean_abs']:.6f} "
                f"max_abs={stats['max_abs']:.6f} "
                f"rmse={stats['rmse']:.6f} "
                f"cosine={stats['cosine_similarity']:.6f}"
            )
    print(f"saved_json: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
