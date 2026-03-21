#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asyncvla_pi import (
    HailoEdgeRunner,
    HailoEdgeRunnerConfig,
    TorchEdgeRunner,
    TorchEdgeRunnerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TorchEdgeRunner and HailoEdgeRunner outputs for the same inputs."
    )
    parser.add_argument("--hf-dir", default="~/gitrepo/AsyncVLA_release")
    parser.add_argument("--hef", default="models/edge_adapter_v520.hef")
    parser.add_argument("--current-image", default=None, help="RGB current image path (png/jpg).")
    parser.add_argument("--delayed-image", default=None, help="RGB delayed/past image path (png/jpg).")
    parser.add_argument("--projected-tokens", default=None, help="Optional .npy path for projected tokens.")
    parser.add_argument("--goal-pose", nargs="*", type=float, default=None, help="Optional goal pose values.")
    parser.add_argument("--synthetic-width", type=int, default=640)
    parser.add_argument("--synthetic-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    # Common preprocessing flags for both runners.
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--normalize-imagenet", action="store_true")
    parser.add_argument("--image-scale-255", action="store_true")

    # Hailo HEF I/O settings.
    parser.add_argument("--input-current-name", default="edge/input_layer1")
    parser.add_argument("--input-delayed-name", default="edge/input_layer2")
    parser.add_argument("--input-tokens-name", default="edge/input_layer3")
    parser.add_argument("--input-goal-name", default=None)
    parser.add_argument("--output-chunk-name", default="edge/depth_to_space1")
    parser.add_argument("--image-layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--input-format", choices=["uint8", "int8", "float32", "auto"], default="uint8")
    parser.add_argument("--output-format", choices=["uint8", "int8", "float32", "auto"], default="auto")

    parser.add_argument("--allclose-rtol", type=float, default=1e-2)
    parser.add_argument("--allclose-atol", type=float, default=1e-2)
    parser.add_argument("--save-npz", default="", help="Optional path to save inputs/outputs/report as .npz.")
    return parser.parse_args()


def _load_image(
    path: str | None,
    *,
    width: int,
    height: int,
    seed: int,
    label: str,
) -> tuple[np.ndarray, str]:
    if path is None:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        return image, f"synthetic_{label}(seed={seed}, shape={image.shape})"

    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    return image, str(image_path)


def _load_projected_tokens(
    path: str | None,
    *,
    chunk_size: int,
    token_dim: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if path is None:
        rng = np.random.default_rng(seed)
        tokens = rng.standard_normal((chunk_size, token_dim)).astype(np.float32)
        return tokens, f"synthetic_tokens(seed={seed}, shape={tokens.shape})"

    tokens_path = Path(path).expanduser().resolve()
    if not tokens_path.exists():
        raise FileNotFoundError(f"Projected tokens file not found: {tokens_path}")
    tokens = np.load(tokens_path)
    return np.asarray(tokens, dtype=np.float32), str(tokens_path)


def _summarize_array(name: str, value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.float32)
    print(
        f"{name}: shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={float(arr.min()):.6f} max={float(arr.max()):.6f} "
        f"mean={float(arr.mean()):.6f} std={float(arr.std()):.6f}"
    )


def _compare_outputs(
    torch_out: np.ndarray,
    hailo_out: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    torch_arr = np.asarray(torch_out, dtype=np.float32)
    hailo_arr = np.asarray(hailo_out, dtype=np.float32)
    if torch_arr.shape != hailo_arr.shape:
        raise ValueError(
            f"Output shape mismatch: torch={tuple(torch_arr.shape)} hailo={tuple(hailo_arr.shape)}"
        )

    diff = torch_arr - hailo_arr
    torch_flat = torch_arr.reshape(-1)
    hailo_flat = hailo_arr.reshape(-1)
    denom = float(np.linalg.norm(torch_flat) * np.linalg.norm(hailo_flat))
    cosine = float(np.dot(torch_flat, hailo_flat) / denom) if denom > 0.0 else 1.0
    return {
        "shape": list(torch_arr.shape),
        "max_abs": float(np.abs(diff).max()),
        "mean_abs": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_signed": float(diff.mean()),
        "cosine_similarity": cosine,
        "allclose": bool(np.allclose(torch_arr, hailo_arr, rtol=rtol, atol=atol)),
        "allclose_rtol": float(rtol),
        "allclose_atol": float(atol),
    }


def _save_report_npz(
    save_path: str,
    *,
    current_image: np.ndarray,
    delayed_image: np.ndarray,
    projected_tokens: np.ndarray,
    goal_pose: np.ndarray | None,
    torch_out: np.ndarray,
    hailo_out: np.ndarray,
    report: dict[str, Any],
) -> Path:
    out_path = Path(save_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        current_image=current_image,
        delayed_image=delayed_image,
        projected_tokens=projected_tokens,
        goal_pose=np.asarray([] if goal_pose is None else goal_pose, dtype=np.float32),
        torch_out=np.asarray(torch_out, dtype=np.float32),
        hailo_out=np.asarray(hailo_out, dtype=np.float32),
        report_json=json.dumps(report, indent=2, sort_keys=True),
    )
    return out_path


def main() -> int:
    args = parse_args()

    torch_runner = TorchEdgeRunner(
        TorchEdgeRunnerConfig(
            hf_dir=args.hf_dir,
            image_height=96,
            image_width=96,
            normalize_imagenet=args.normalize_imagenet,
            image_scale_255=args.image_scale_255,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
            device=args.torch_device,
            dtype=args.torch_dtype,
        )
    )

    hailo_runner = HailoEdgeRunner(
        HailoEdgeRunnerConfig(
            hef_path=args.hef,
            input_current_image=args.input_current_name,
            input_delayed_image=args.input_delayed_name,
            input_projected_tokens=args.input_tokens_name,
            input_goal_pose=args.input_goal_name,
            output_action_chunk=args.output_chunk_name,
            image_height=96,
            image_width=96,
            chunk_size=int(torch_runner.model.action_chunk_size),
            pose_dim=int(torch_runner.model.action_dim),
            image_layout=args.image_layout,
            input_format_type=args.input_format,
            output_format_type=args.output_format,
            normalize_imagenet=args.normalize_imagenet,
            image_scale_255=args.image_scale_255,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )
    )

    try:
        current_image, current_source = _load_image(
            args.current_image,
            width=args.synthetic_width,
            height=args.synthetic_height,
            seed=args.seed,
            label="current_image",
        )
        delayed_seed = args.seed + 1 if args.delayed_image is None else args.seed
        delayed_image, delayed_source = _load_image(
            args.delayed_image,
            width=args.synthetic_width,
            height=args.synthetic_height,
            seed=delayed_seed,
            label="delayed_image",
        )
        projected_tokens, token_source = _load_projected_tokens(
            args.projected_tokens,
            chunk_size=int(torch_runner.model.action_chunk_size),
            token_dim=int(torch_runner.model.obs_encoding_size),
            seed=args.seed + 2,
        )
        goal_pose = None if args.goal_pose is None else np.asarray(args.goal_pose, dtype=np.float32)

        print(f"hf_dir: {Path(args.hf_dir).expanduser().resolve()}")
        print(f"hef: {Path(args.hef).expanduser().resolve()}")
        print(f"current_image: {current_source}")
        print(f"delayed_image: {delayed_source}")
        print(f"projected_tokens: {token_source}")
        print(
            "preprocess: "
            f"convert_bgr_to_rgb={args.convert_bgr_to_rgb} "
            f"normalize_imagenet={args.normalize_imagenet} "
            f"image_scale_255={args.image_scale_255}"
        )
        print(
            "hailo_io: "
            f"image_layout={args.image_layout} input_format={args.input_format} output_format={args.output_format}"
        )
        if goal_pose is not None:
            print(f"goal_pose: {goal_pose.tolist()}")

        _summarize_array("current_image", current_image)
        _summarize_array("delayed_image", delayed_image)
        _summarize_array("projected_tokens", projected_tokens)

        torch_out = torch_runner.infer(
            current_image=current_image,
            delayed_image=delayed_image,
            projected_tokens=projected_tokens,
            goal_pose=goal_pose,
        )
        hailo_out = hailo_runner.infer(
            current_image=current_image,
            delayed_image=delayed_image,
            projected_tokens=projected_tokens,
            goal_pose=goal_pose,
        )
        report = _compare_outputs(
            torch_out,
            hailo_out,
            rtol=args.allclose_rtol,
            atol=args.allclose_atol,
        )

        _summarize_array("torch_out", torch_out)
        _summarize_array("hailo_out", hailo_out)
        print("diff_report:")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("torch_out_sample:", np.asarray(torch_out, dtype=np.float32).reshape(-1)[:8].tolist())
        print("hailo_out_sample:", np.asarray(hailo_out, dtype=np.float32).reshape(-1)[:8].tolist())

        if args.save_npz:
            saved = _save_report_npz(
                args.save_npz,
                current_image=current_image,
                delayed_image=delayed_image,
                projected_tokens=projected_tokens,
                goal_pose=goal_pose,
                torch_out=torch_out,
                hailo_out=hailo_out,
                report=report,
            )
            print(f"saved_npz: {saved}")
        return 0
    finally:
        torch_runner.close()
        hailo_runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
