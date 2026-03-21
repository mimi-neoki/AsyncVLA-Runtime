#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build calibration samples from local LeLaN dataset subset."
    )
    parser.add_argument(
        "--lelan-repo",
        default="~/gitrepo/learning-language-navigation",
        help="Path to learning-language-navigation repository.",
    )
    parser.add_argument(
        "--config",
        default="train/config/lelan.yaml",
        help="Path (relative to lelan-repo or absolute) to LeLaN yaml config.",
    )
    parser.add_argument("--output-dir", default="calib_data")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--skip-no-prompt", action="store_true", default=True)
    parser.add_argument(
        "--exclude-root-keywords",
        default="youtube",
        help="Comma-separated keywords. Samples from source paths containing these tokens are skipped.",
    )
    parser.add_argument(
        "--exclude-prompt-terms",
        default="car,truck,bus,motorcycle,bicycle,road,street,traffic,highway,sky,mountain,beach,forest",
        help="Comma-separated terms. Prompts containing these terms are skipped.",
    )
    parser.add_argument(
        "--require-prompt-terms",
        default="door,chair,table,desk,sofa,cabinet,bed,room,hallway,corridor,stairs,shelf,monitor,laptop,tv,fridge,microwave,sink,toilet,kitchen,bathroom,office",
        help="Comma-separated terms. Prompt must contain at least one to be included.",
    )
    parser.add_argument(
        "--policy-url",
        default="",
        help="If set, collect projected_tokens by posting each sample to base-VLA server /infer endpoint.",
    )
    parser.add_argument(
        "--policy-image-key",
        default="front_image",
        help="Image key expected by policy server payload.",
    )
    parser.add_argument(
        "--policy-timeout-sec",
        type=float,
        default=30.0,
        help="HTTP timeout for each policy request.",
    )
    parser.add_argument(
        "--policy-retries",
        type=int,
        default=2,
        help="Retries per sample when policy request fails.",
    )
    return parser.parse_args()


def _resolve_config_path(lelan_repo: Path, config_arg: str) -> Path:
    cfg = Path(config_arg).expanduser()
    if cfg.is_absolute():
        return cfg
    return (lelan_repo / cfg).resolve()


def _load_image_roots_from_yaml(config_path: Path, lelan_repo: Path) -> list[Path]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = data.get("datasets", {})
    roots: list[Path] = []
    base_cfg = config_path.parent
    base_train = config_path.parent.parent
    for _, cfg in datasets.items():
        image_rel = cfg.get("image")
        if not image_rel:
            continue
        rel_path = Path(str(image_rel)).expanduser()
        if rel_path.is_absolute():
            roots.append(rel_path.resolve())
            continue
        roots.append((base_cfg / rel_path).resolve())
        roots.append((base_train / rel_path).resolve())
        roots.append((lelan_repo / rel_path).resolve())
    # Unique, keep order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _discover_image_roots_from_repo_layout(lelan_repo: Path) -> list[Path]:
    roots: list[Path] = []
    bundle_root = lelan_repo / "dataset_LeLaN"
    if not bundle_root.exists():
        return roots
    for dataset_dir in sorted(bundle_root.glob("dataset_LeLaN_*")):
        if not dataset_dir.is_dir():
            continue
        direct_image = dataset_dir / "image"
        if direct_image.exists():
            roots.append(direct_image.resolve())
        else:
            roots.append(dataset_dir.resolve())
    return roots


def _find_images(image_root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: list[Path] = []
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def _candidate_pickle_paths(image_path: Path) -> list[Path]:
    candidates: list[Path] = []
    stem = image_path.stem

    # same directory
    candidates.append(image_path.with_suffix(".pkl"))

    # sibling pickle directory near image directory
    candidates.append(image_path.parent / f"{stem}.pkl")
    candidates.append(image_path.parent.parent / "pickle" / f"{stem}.pkl")

    s = str(image_path)
    for token in ["/image/", "/images/", "\\image\\", "\\images\\"]:
        if token in s:
            candidates.append(Path(s.replace(token, token.replace("image", "pickle").replace("images", "pickle"))).with_suffix(".pkl"))
            break

    # de-dup order preserving
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in candidates:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _prompt_from_pickle(pickle_path: Path) -> str | None:
    if not pickle_path.exists() or pickle_path.stat().st_size == 0:
        return None
    try:
        with pickle_path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return None

    entries: list[dict[str, Any]] = []
    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, (list, tuple)):
        entries = [x for x in data if isinstance(x, dict)]

    best: tuple[int, str] | None = None
    for e in entries:
        prompt_obj = e.get("prompt")
        prompt: str | None = None
        if isinstance(prompt_obj, str):
            txt = " ".join(prompt_obj.strip().split())
            prompt = txt if txt else None
        elif isinstance(prompt_obj, (list, tuple)):
            for x in prompt_obj:
                if isinstance(x, str):
                    txt = " ".join(x.strip().split())
                    if txt:
                        prompt = txt
                        break
        if not prompt:
            continue
        score = 0
        if bool(e.get("obj_detect", False)):
            score += 2
        if "bbox" in e:
            score += 1
        if best is None or score > best[0]:
            best = (score, prompt)
    return None if best is None else best[1]


def _normalize_instruction(prompt: str | None) -> str:
    if not prompt:
        return "move forward indoors"
    text = " ".join(str(prompt).strip().split())
    if not text:
        return "move forward indoors"
    lower = text.lower()
    if lower.startswith(("move to ", "go to ", "approach ", "find ")):
        return text
    return f"move to {text}"


def _contains_any_term(text: str, terms: list[str]) -> bool:
    t = f" {text.lower()} "
    for term in terms:
        term = term.strip().lower()
        if not term:
            continue
        if f" {term} " in t:
            return True
    return False


def _write_outputs(
    out_dir: Path,
    samples: list[dict[str, Any]],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    arr_u8 = np.empty((len(samples), image_size, image_size, 3), dtype=np.uint8)
    for i, sample in enumerate(samples):
        src = Path(sample["source_image_abs"])
        dst = images_dir / sample["file_name"]
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            src_w, src_h = rgb.size

            # For dual-camera side-by-side images, use left half (front camera).
            crop_applied = False
            if src_w > src_h:
                rgb = rgb.crop((0, 0, src_w // 2, src_h))
                crop_applied = True

            proc_w, proc_h = rgb.size
            sample["source_image_size"] = [int(src_w), int(src_h)]
            sample["processed_image_size"] = [int(proc_w), int(proc_h)]
            sample["front_left_crop_applied"] = bool(crop_applied)

            # Save the processed (front-view) image for inspection.
            rgb.save(dst)

            rgb_96 = rgb.resize((image_size, image_size), Image.BILINEAR)
            arr_u8[i] = np.asarray(rgb_96, dtype=np.uint8)

    arr_f32 = arr_u8.astype(np.float32) / 255.0
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_uint8.npy", arr_u8)
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_float32_01.npy", arr_f32)

    (out_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "instructions.txt").open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(item["instruction"] + "\n")

    readme = [
        "# Calibration Data from LeLaN Dataset",
        "",
        f"- Samples: {len(samples)}",
        "- Source: Local LeLaN dataset images + prompt labels from pkl files",
        "",
        "## Files",
        "- `images/*.jpg`: copied images",
        "- `samples.json`, `samples.jsonl`: metadata",
        "- `instructions.txt`: one instruction per line",
        f"- `calib_images_{image_size}x{image_size}_uint8.npy`",
        f"- `calib_images_{image_size}x{image_size}_float32_01.npy`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return arr_u8, arr_f32


def _post_policy_infer(
    policy_url: str,
    image_key: str,
    instruction: str,
    image_u8: np.ndarray,
    timeout_sec: float,
) -> np.ndarray:
    payload = {
        "timestamp_ns": int(time.monotonic_ns()),
        "instruction": instruction,
        "images": {
            image_key: {
                "data": image_u8.tolist(),
                "shape": list(image_u8.shape),
            }
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        policy_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        resp_body = resp.read()
    result = json.loads(resp_body.decode("utf-8"))
    if "projected_tokens" not in result:
        raise RuntimeError(f"Missing projected_tokens in response: keys={list(result.keys())}")
    tokens = np.asarray(result["projected_tokens"], dtype=np.float32)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 2:
        raise RuntimeError(f"Unexpected projected_tokens shape: {tokens.shape}")
    return tokens


def _collect_projected_tokens(
    out_dir: Path,
    samples: list[dict[str, Any]],
    arr_u8: np.ndarray,
    arr_f32: np.ndarray,
    image_size: int,
    policy_url: str,
    policy_image_key: str,
    timeout_sec: float,
    retries: int,
) -> None:
    if arr_u8.shape[0] != len(samples):
        raise RuntimeError("arr_u8 and samples length mismatch")

    tokens_list: list[np.ndarray] = []
    for i, sample in enumerate(samples):
        instruction = str(sample.get("instruction", "")).strip()
        if not instruction:
            instruction = "move forward indoors"

        last_err: Exception | None = None
        for _ in range(max(0, retries) + 1):
            try:
                tok = _post_policy_infer(
                    policy_url=policy_url,
                    image_key=policy_image_key,
                    instruction=instruction,
                    image_u8=arr_u8[i],
                    timeout_sec=timeout_sec,
                )
                tokens_list.append(tok)
                sample["projected_token_shape"] = list(tok.shape)
                break
            except Exception as exc:  # noqa: PERF203
                last_err = exc
                time.sleep(0.2)
        else:
            raise RuntimeError(f"Failed to fetch projected_tokens for sample={i}: {last_err}") from last_err

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"[tokens] collected {i + 1}/{len(samples)}")

    projected = np.stack(tokens_list, axis=0).astype(np.float32, copy=False)  # [N, 8, 1024]
    projected_n1 = projected[:, None, :, :]  # [N, 1, 8, 1024]
    delayed = np.roll(arr_f32, shift=1, axis=0)
    delayed_u8 = np.roll(arr_u8, shift=1, axis=0)

    np.save(out_dir / "calib_projected_tokens_8x1024_float32.npy", projected)
    np.save(out_dir / "calib_projected_tokens_1x8x1024_float32.npy", projected_n1)
    np.savez(
        out_dir / f"edge_adapter_calib_inputs_{image_size}x{image_size}_n{arr_u8.shape[0]}.npz",
        current_image=arr_f32,
        delayed_image=delayed,
        projected_tokens=projected,
        **{
            "edge_adapter_static/input_layer1": arr_f32,
            "edge_adapter_static/input_layer2": delayed,
            "edge_adapter_static/input_layer3": projected_n1,
        },
    )
    np.savez(
        out_dir / f"edge_adapter_calib_inputs_hailo_only_u8img_n{arr_u8.shape[0]}.npz",
        current_image=arr_u8,
        delayed_image=delayed_u8,
        projected_tokens=projected,
        **{
            "edge_adapter_static/input_layer1": arr_u8,
            "edge_adapter_static/input_layer2": delayed_u8,
            "edge_adapter_static/input_layer3": projected_n1,
        },
    )

    readme_path = out_dir / "README.md"
    lines = readme_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    extra = [
        "",
        "## Projected Tokens",
        f"- policy_url: `{policy_url}`",
        f"- `calib_projected_tokens_8x1024_float32.npy`: shape [N, 8, 1024]",
        f"- `calib_projected_tokens_1x8x1024_float32.npy`: shape [N, 1, 8, 1024]",
        f"- `edge_adapter_calib_inputs_{image_size}x{image_size}_n{arr_u8.shape[0]}.npz`: multi-input calibration pack",
        f"- `edge_adapter_calib_inputs_hailo_only_u8img_n{arr_u8.shape[0]}.npz`: multi-input calibration pack with uint8 image tensors for Hailo calibration",
    ]
    readme_path.write_text("\n".join(lines + extra) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    lelan_repo = Path(args.lelan_repo).expanduser().resolve()
    config_path = _resolve_config_path(lelan_repo=lelan_repo, config_arg=str(args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"LeLaN config not found: {config_path}")

    image_roots_all = _load_image_roots_from_yaml(config_path=config_path, lelan_repo=lelan_repo)
    image_roots = [p for p in image_roots_all if p.exists()]
    if not image_roots:
        fallback_roots = _discover_image_roots_from_repo_layout(lelan_repo=lelan_repo)
        image_roots_all.extend(fallback_roots)
        image_roots = [p for p in image_roots_all if p.exists()]
    if not image_roots:
        expected = "\n".join(f"- {p}" for p in image_roots_all)
        raise FileNotFoundError(
            "No LeLaN image roots found on disk. Download/unzip LeLaN dataset first.\n"
            f"Checked:\n{expected}"
        )

    exclude_root_terms = [x.strip().lower() for x in str(args.exclude_root_keywords).split(",") if x.strip()]
    exclude_prompt_terms = [x.strip().lower() for x in str(args.exclude_prompt_terms).split(",") if x.strip()]
    require_prompt_terms = [x.strip().lower() for x in str(args.require_prompt_terms).split(",") if x.strip()]

    pool: list[dict[str, Any]] = []
    for root in image_roots:
        root_lower = str(root).lower()
        if exclude_root_terms and any(term in root_lower for term in exclude_root_terms):
            continue
        imgs = _find_images(root)
        for img in imgs:
            pkl_path = None
            for cand in _candidate_pickle_paths(img):
                if cand.exists():
                    pkl_path = cand
                    break
            prompt = _prompt_from_pickle(pkl_path) if pkl_path is not None else None
            if args.skip_no_prompt and not prompt:
                continue
            if prompt and _contains_any_term(prompt, exclude_prompt_terms):
                continue
            if prompt and require_prompt_terms and (not _contains_any_term(prompt, require_prompt_terms)):
                continue
            instruction = _normalize_instruction(prompt)
            pool.append(
                {
                    "source_image_abs": str(img.resolve()),
                    "source_pickle_abs": None if pkl_path is None else str(pkl_path.resolve()),
                    "source_root": str(root),
                    "instruction": instruction,
                }
            )

    if not pool:
        raise RuntimeError("No usable LeLaN samples found (prompted samples are empty).")

    out_dir = Path(args.output_dir).expanduser().resolve()
    if args.clean_output and out_dir.exists():
        for name in [
            "images",
            "samples.json",
            "samples.jsonl",
            "instructions.txt",
            "README.md",
            f"calib_images_{args.image_size}x{args.image_size}_uint8.npy",
            f"calib_images_{args.image_size}x{args.image_size}_float32_01.npy",
        ]:
            path = out_dir / name
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    rng.shuffle(pool)

    n = min(int(args.num_samples), len(pool))
    selected = pool[:n]
    for i, item in enumerate(selected):
        suffix = Path(item["source_image_abs"]).suffix.lower()
        item["sample_index"] = i
        item["file_name"] = f"{i:04d}{suffix if suffix in {'.jpg', '.jpeg', '.png'} else '.jpg'}"
        item["local_image_path"] = f"images/{item['file_name']}"

    arr_u8, arr_f32 = _write_outputs(out_dir=out_dir, samples=selected, image_size=int(args.image_size))
    if str(args.policy_url).strip():
        _collect_projected_tokens(
            out_dir=out_dir,
            samples=selected,
            arr_u8=arr_u8,
            arr_f32=arr_f32,
            image_size=int(args.image_size),
            policy_url=str(args.policy_url).strip(),
            policy_image_key=str(args.policy_image_key),
            timeout_sec=float(args.policy_timeout_sec),
            retries=int(args.policy_retries),
        )

    print("[done] LeLaN calibration set generated")
    print(f"[done] output_dir={out_dir}")
    print(f"[done] selected={len(selected)} pool={len(pool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
