#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image


OPENVERSE_API = "https://api.openverse.org/v1/images/"

DEFAULT_QUERIES = [
    "indoor hallway floor",
    "indoor corridor floor",
    "interior low angle room",
    "ground level indoor room",
    "floor perspective hallway",
    "kitchen interior floor",
    "living room interior floor",
    "office hallway interior",
    "apartment corridor interior",
    "hotel hallway interior",
    "indoor stairs floor level",
    "indoor doorway corridor",
    "first person indoor hallway",
    "interior room from floor",
    "indoor navigation corridor",
]

INDOOR_TERMS = {
    "indoor",
    "indoors",
    "interior",
    "hallway",
    "corridor",
    "room",
    "kitchen",
    "living",
    "office",
    "apartment",
    "hotel",
    "bathroom",
    "bedroom",
    "lobby",
    "stair",
    "stairs",
    "doorway",
}

FLOOR_LEVEL_TERMS = {
    "floor",
    "ground",
    "low",
    "low-angle",
    "lowangle",
    "perspective",
    "first-person",
    "firstperson",
    "from floor",
}

OUTDOOR_TERMS = {
    "outdoor",
    "outdoors",
    "street",
    "road",
    "sky",
    "mountain",
    "beach",
    "forest",
    "landscape",
    "field",
    "car",
    "highway",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect indoor floor-level calibration samples for navigation VLA."
    )
    parser.add_argument("--output-dir", default="calib_data")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-pages-per-query", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=5.0)
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _score_candidate(item: dict[str, Any]) -> float:
    title = _normalize_text(str(item.get("title", "")))
    tag_objs = item.get("tags", []) or []
    tags = [_normalize_text(str(t.get("name", ""))) for t in tag_objs]
    tags_text = " ".join(tags)
    text = f"{title} {tags_text}"

    score = 0.0
    for term in INDOOR_TERMS:
        if term in text:
            score += 1.2
    for term in FLOOR_LEVEL_TERMS:
        if term in text:
            score += 1.8
    for term in OUTDOOR_TERMS:
        if term in text:
            score -= 2.0

    # Additional bonus for model tags.
    for t in tag_objs:
        name = _normalize_text(str(t.get("name", "")))
        acc = float(t.get("accuracy", 0.0) or 0.0)
        if name in {"indoors", "hallway", "corridor", "floor"}:
            score += 2.5 * acc
        if name in {"outdoors", "street", "road"}:
            score -= 2.5 * acc
    return score


def _instruction_from_item(item: dict[str, Any]) -> str:
    tags = [_normalize_text(str(t.get("name", ""))) for t in (item.get("tags") or [])]
    text = " ".join(tags)
    if "corridor" in text or "hallway" in text:
        return "move forward through the hallway"
    if "doorway" in text or "door" in text:
        return "move toward the doorway"
    if "stairs" in text or "stair" in text:
        return "approach the stairs"
    if "kitchen" in text:
        return "move toward the kitchen area"
    if "living room" in text or "livingroom" in text:
        return "move toward the living room"
    if "office" in text:
        return "move toward the office corridor"
    title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
    if title:
        return f"move to {title}"
    return "move forward indoors"


def _request_json(url: str, params: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _collect_candidates(
    queries: list[str],
    page_size: int,
    max_pages_per_query: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates_by_id: dict[str, dict[str, Any]] = {}
    error_count = 0
    for query in queries:
        for page in range(1, max_pages_per_query + 1):
            params = {"q": query, "page_size": page_size, "page": page}
            try:
                body = _request_json(OPENVERSE_API, params=params)
            except Exception:
                error_count += 1
                if error_count <= 5:
                    print(f"[warn] query failed: q='{query}' page={page}")
                break
            results = body.get("results", []) or []
            if not results:
                break
            for item in results:
                item_id = str(item.get("id", ""))
                url = str(item.get("url", ""))
                if (not item_id) or (not url):
                    continue
                record = dict(item)
                record["query"] = query
                record["score"] = _score_candidate(record)
                if item_id not in candidates_by_id or record["score"] > candidates_by_id[item_id]["score"]:
                    candidates_by_id[item_id] = record
            # Be polite to API.
            time.sleep(0.08)
    candidates = list(candidates_by_id.values())
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return candidates


def _download_image(url: str, out_path: Path, timeout_s: float = 30.0, retries: int = 3) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout_s) as resp:
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            with Image.open(out_path) as im:
                im.verify()
            return True
        except Exception:
            out_path.unlink(missing_ok=True)
            if i + 1 < retries:
                time.sleep(0.35 * (i + 1))
    return False


def _valid_dims(item: dict[str, Any], min_w: int, min_h: int) -> bool:
    w = int(item.get("width") or 0)
    h = int(item.get("height") or 0)
    return w >= min_w and h >= min_h


def _write_outputs(
    out_dir: Path,
    selected: list[dict[str, Any]],
    image_size: int,
) -> None:
    images_dir = out_dir / "images"
    samples_json = out_dir / "samples.json"
    samples_jsonl = out_dir / "samples.jsonl"
    instructions_txt = out_dir / "instructions.txt"

    samples_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    with samples_jsonl.open("w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with instructions_txt.open("w", encoding="utf-8") as f:
        for sample in selected:
            f.write(sample["instruction"] + "\n")

    n = len(selected)
    arr_u8 = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    for i, sample in enumerate(selected):
        image_path = images_dir / sample["file_name"]
        with Image.open(image_path) as im:
            rgb = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr_u8[i] = np.asarray(rgb, dtype=np.uint8)
    arr_f32 = arr_u8.astype(np.float32) / 255.0
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_uint8.npy", arr_u8)
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_float32_01.npy", arr_f32)

    readme_lines = [
        "# Navigation Calibration Data",
        "",
        "- Focus: indoor, near-floor / hallway / corridor viewpoint images",
        f"- Samples: {len(selected)}",
        "- Source: Openverse image API",
        "- Each record stores source URL, license, and attribution metadata.",
        "",
        "## Files",
        "- `images/*.jpg`: downloaded images",
        "- `samples.json` and `samples.jsonl`: sample metadata",
        "- `instructions.txt`: one navigation instruction per sample",
        f"- `calib_images_{image_size}x{image_size}_uint8.npy`",
        f"- `calib_images_{image_size}x{image_size}_float32_01.npy`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = out_dir / "images"

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
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[collect] querying Openverse candidates...")
    candidates = _collect_candidates(
        queries=DEFAULT_QUERIES,
        page_size=int(args.page_size),
        max_pages_per_query=int(args.max_pages_per_query),
        seed=int(args.seed),
    )
    print(f"[collect] candidates_total={len(candidates)}")

    min_score = float(args.min_score)
    filtered = [c for c in candidates if float(c.get("score", 0.0)) >= min_score]
    print(f"[collect] candidates_score>={min_score:.1f}: {len(filtered)}")

    target_n = int(args.num_samples)
    selected: list[dict[str, Any]] = []
    used_urls: set[str] = set()
    failures = 0
    for idx, item in enumerate(filtered):
        if len(selected) >= target_n:
            break
        if not _valid_dims(item, min_w=int(args.min_width), min_h=int(args.min_height)):
            continue
        url = str(item.get("url", ""))
        if not url or url in used_urls:
            continue

        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        file_name = f"{len(selected):04d}_{str(item.get('id', 'img')).replace('/', '_')[:32]}{ext}"
        out_path = images_dir / file_name
        ok = _download_image(url, out_path)
        if not ok:
            failures += 1
            continue

        record = {
            "sample_index": len(selected),
            "file_name": file_name,
            "local_image_path": f"images/{file_name}",
            "source_url": url,
            "foreign_landing_url": item.get("foreign_landing_url"),
            "title": item.get("title"),
            "provider": item.get("provider"),
            "source": item.get("source"),
            "creator": item.get("creator"),
            "license": item.get("license"),
            "license_version": item.get("license_version"),
            "license_url": item.get("license_url"),
            "width": item.get("width"),
            "height": item.get("height"),
            "query": item.get("query"),
            "score": float(item.get("score", 0.0)),
            "tags": [t.get("name") for t in (item.get("tags") or [])][:24],
            "instruction": _instruction_from_item(item),
        }
        selected.append(record)
        used_urls.add(url)
        if len(selected) % 50 == 0:
            print(f"[progress] downloaded {len(selected)}/{target_n} (failures={failures}, scanned={idx+1})")

    if len(selected) < target_n:
        raise RuntimeError(
            f"Not enough selected samples: got={len(selected)} target={target_n} failures={failures}. "
            f"Try lowering --min-score or increasing --max-pages-per-query."
        )

    _write_outputs(out_dir=out_dir, selected=selected, image_size=int(args.image_size))
    print("[done] generated calibration set")
    print(f"[done] output_dir={out_dir}")
    print(f"[done] samples={len(selected)} failures={failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
