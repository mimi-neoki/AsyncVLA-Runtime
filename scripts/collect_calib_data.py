#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image


COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"
COCO_CAPTIONS_JSON_IN_ZIP = "annotations/captions_val2017.json"


def _download_file(url: str, out_path: Path, timeout_s: float = 30.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _ensure_coco_captions_json(cache_dir: Path) -> Path:
    ann_zip = cache_dir / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        print(f"[download] {COCO_ANN_ZIP_URL} -> {ann_zip}")
        _download_file(COCO_ANN_ZIP_URL, ann_zip, timeout_s=120.0)
    out_json = cache_dir / "captions_val2017.json"
    if not out_json.exists():
        print(f"[extract] {COCO_CAPTIONS_JSON_IN_ZIP} -> {out_json}")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            with zf.open(COCO_CAPTIONS_JSON_IN_ZIP, "r") as src:
                out_json.write_bytes(src.read())
    return out_json


def _build_instruction(caption: str) -> str:
    text = " ".join(caption.strip().split())
    if text.endswith("."):
        text = text[:-1].strip()
    if not text:
        return "move to the target object"
    lower = text.lower()
    if lower.startswith(("move to ", "go to ", "approach ", "find ")):
        return text
    return f"move to {text}"


def _load_coco_records(captions_json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(captions_json_path.read_text(encoding="utf-8"))
    image_by_id: dict[int, dict[str, Any]] = {}
    for image in payload["images"]:
        image_by_id[int(image["id"])] = image

    captions_by_id: dict[int, list[str]] = defaultdict(list)
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        caption = str(ann["caption"])
        captions_by_id[image_id].append(caption)

    records: list[dict[str, Any]] = []
    for image_id, image_info in image_by_id.items():
        caps = captions_by_id.get(image_id)
        if not caps:
            continue
        # Prefer slightly richer caption to diversify noun phrases.
        caption = sorted(caps, key=lambda x: (len(x), x), reverse=True)[0]
        file_name = str(image_info["file_name"])
        image_url = f"{COCO_VAL_IMAGE_BASE_URL}/{file_name}"
        records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_url": image_url,
                "caption": caption,
                "instruction": _build_instruction(caption),
                "source": "MS COCO 2017 val",
            }
        )
    return records


def _download_image(url: str, out_path: Path, retries: int = 3, timeout_s: float = 30.0) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            _download_file(url, out_path, timeout_s=timeout_s)
            # Validate image file quickly.
            with Image.open(out_path) as im:
                im.verify()
            return True
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            if attempt == retries:
                return False
            time.sleep(0.3 * attempt)
    return False


def _build_calib_arrays(
    samples: list[dict[str, Any]],
    images_dir: Path,
    out_uint8_npy: Path,
    out_float_npy: Path,
    image_size: int,
) -> None:
    n = len(samples)
    arr_u8 = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    for i, sample in enumerate(samples):
        image_path = images_dir / sample["file_name"]
        with Image.open(image_path) as im:
            rgb = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr_u8[i] = np.asarray(rgb, dtype=np.uint8)
    arr_f32 = arr_u8.astype(np.float32) / 255.0
    np.save(out_uint8_npy, arr_u8)
    np.save(out_float_npy, arr_f32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect calibration data from internet (COCO val2017).")
    parser.add_argument("--output-dir", default="calib_data")
    parser.add_argument("--cache-dir", default="calib_data/.cache")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    captions_json = _ensure_coco_captions_json(cache_dir)
    candidates = _load_coco_records(captions_json)
    rnd = random.Random(int(args.seed))
    rnd.shuffle(candidates)

    target_n = int(args.num_samples)
    selected: list[dict[str, Any]] = []
    failures = 0
    for rec in candidates:
        if len(selected) >= target_n:
            break
        image_path = images_dir / rec["file_name"]
        if image_path.exists():
            ok = True
        else:
            ok = _download_image(rec["image_url"], image_path)
        if not ok:
            failures += 1
            continue
        rec = dict(rec)
        rec["local_image_path"] = str(image_path.relative_to(out_dir))
        rec["sample_index"] = len(selected)
        selected.append(rec)
        if len(selected) % 50 == 0:
            print(f"[progress] collected {len(selected)}/{target_n}")

    if len(selected) < target_n:
        raise RuntimeError(
            f"Could not collect enough samples. collected={len(selected)} target={target_n} failures={failures}"
        )

    manifest_json = out_dir / "samples.json"
    manifest_jsonl = out_dir / "samples.jsonl"
    instructions_txt = out_dir / "instructions.txt"
    manifest_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with instructions_txt.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(item["instruction"] + "\n")

    calib_u8_path = out_dir / f"calib_images_{args.image_size}x{args.image_size}_uint8.npy"
    calib_f32_path = out_dir / f"calib_images_{args.image_size}x{args.image_size}_float32_01.npy"
    _build_calib_arrays(
        samples=selected,
        images_dir=images_dir,
        out_uint8_npy=calib_u8_path,
        out_float_npy=calib_f32_path,
        image_size=int(args.image_size),
    )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Calibration Data",
                "",
                "- Source: MS COCO 2017 val images/captions (downloaded from `images.cocodataset.org`)",
                f"- Number of samples: {len(selected)}",
                f"- Image size for npy tensors: {args.image_size}x{args.image_size}",
                "",
                "## Files",
                "- `images/*.jpg`: downloaded raw images",
                "- `samples.json` / `samples.jsonl`: metadata including caption, generated instruction, source URL",
                "- `instructions.txt`: one instruction per line (512 lines)",
                "- `calib_images_96x96_uint8.npy`: uint8 NHWC tensor suitable for uint8 calibration pipelines",
                "- `calib_images_96x96_float32_01.npy`: float32 NHWC tensor in [0,1]",
                "",
                "## Note",
                "- This set is for calibration/experimentation and keeps source attribution in manifest files.",
            ]
        ),
        encoding="utf-8",
    )

    print("[done] calib data generated")
    print(f"[done] output_dir={out_dir}")
    print(f"[done] samples={len(selected)} failures={failures}")
    print(f"[done] manifest={manifest_json}")
    print(f"[done] npy_uint8={calib_u8_path}")
    print(f"[done] npy_float32={calib_f32_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
