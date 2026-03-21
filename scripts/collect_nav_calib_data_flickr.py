#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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


FLICKR_FEED_URL = "https://www.flickr.com/services/feeds/photos_public.gne"

INDOOR_TAGS = ["indoor", "interior", "indoors"]
SCENE_TAGS = [
    "hallway",
    "corridor",
    "room",
    "kitchen",
    "livingroom",
    "bedroom",
    "bathroom",
    "office",
    "staircase",
    "doorway",
    "lobby",
    "apartment",
    "hotel",
    "house",
    "home",
    "basement",
]
VIEW_TAGS = [
    "floor",
    "lowangle",
    "low-angle",
    "ground",
    "perspective",
    "firstperson",
    "fromfloor",
]

INDOOR_TERMS = {
    "indoor",
    "indoors",
    "interior",
    "hallway",
    "corridor",
    "room",
    "kitchen",
    "livingroom",
    "bedroom",
    "bathroom",
    "office",
    "staircase",
    "doorway",
    "lobby",
    "apartment",
    "hotel",
    "house",
    "home",
    "basement",
}
FLOOR_TERMS = {
    "floor",
    "lowangle",
    "low-angle",
    "ground",
    "perspective",
    "firstperson",
    "fromfloor",
}
OUTDOOR_TERMS = {
    "outdoor",
    "outdoors",
    "street",
    "road",
    "highway",
    "mountain",
    "beach",
    "forest",
    "sky",
    "landscape",
    "field",
    "garden",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect indoor/floor-near navigation calibration set from Flickr public feed."
    )
    parser.add_argument("--output-dir", default="calib_data")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--min-score", type=float, default=4.5)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--max-queries", type=int, default=260)
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_terms(item: dict[str, Any]) -> set[str]:
    title = _normalize_text(str(item.get("title", "")))
    tags = _normalize_text(str(item.get("tags", "")))
    raw = f"{title} {tags}"
    return set(re.findall(r"[a-z0-9\\-]+", raw))


def _score_item(item: dict[str, Any]) -> float:
    terms = _extract_terms(item)
    score = 0.0
    score += 1.3 * sum(1 for t in INDOOR_TERMS if t in terms)
    score += 2.0 * sum(1 for t in FLOOR_TERMS if t in terms)
    score -= 2.2 * sum(1 for t in OUTDOOR_TERMS if t in terms)
    if ("hallway" in terms) or ("corridor" in terms):
        score += 2.0
    if "floor" in terms:
        score += 2.0
    return score


def _instruction_from_item(item: dict[str, Any]) -> str:
    terms = _extract_terms(item)
    if "hallway" in terms or "corridor" in terms:
        return "move forward through the hallway"
    if "doorway" in terms or "door" in terms:
        return "move toward the doorway"
    if "staircase" in terms or "stairs" in terms:
        return "approach the stairs"
    if "kitchen" in terms:
        return "move toward the kitchen area"
    if "livingroom" in terms:
        return "move toward the living room"
    if "office" in terms:
        return "move toward the office corridor"
    title = " ".join(str(item.get("title", "")).strip().split())
    if title:
        return f"move to {title}"
    return "move forward indoors"


def _query_feed(tags: list[str], tagmode: str = "all", timeout_s: float = 30.0) -> list[dict[str, Any]]:
    params = {
        "format": "json",
        "nojsoncallback": 1,
        "tags": ",".join(tags),
        "tagmode": tagmode,
    }
    resp = requests.get(FLICKR_FEED_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    return body.get("items", []) or []


def _image_candidates_from_media_url(media_url: str) -> list[str]:
    # Flickr suffix candidates: b(large), c, z, n, m(small)
    if "_m." not in media_url:
        return [media_url]
    out: list[str] = []
    for suffix in ["_b.", "_c.", "_z.", "_n.", "_m."]:
        out.append(media_url.replace("_m.", suffix))
    # preserve order, unique
    return list(dict.fromkeys(out))


def _download_image(urls: list[str], out_path: Path, timeout_s: float = 30.0) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for u in urls:
        try:
            with requests.get(u, stream=True, timeout=timeout_s) as resp:
                if resp.status_code != 200:
                    continue
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            with Image.open(out_path) as im:
                im.verify()
            return True, u
        except Exception:
            out_path.unlink(missing_ok=True)
            continue
    return False, ""


def _build_queries(seed: int, max_queries: int) -> list[tuple[list[str], str]]:
    rng = random.Random(seed)
    queries: list[tuple[list[str], str]] = []

    # Strong indoor + scene + floor-level constraints.
    for a, b, c in itertools.product(INDOOR_TAGS, SCENE_TAGS, VIEW_TAGS):
        queries.append(([a, b, c], "all"))

    # Backup indoor + scene pairs.
    for a, b in itertools.product(INDOOR_TAGS, SCENE_TAGS):
        queries.append(([a, b], "all"))

    # Additional mixed queries with any mode for diversity.
    for b, c in itertools.product(SCENE_TAGS, VIEW_TAGS):
        queries.append(([b, c, "interior"], "any"))

    rng.shuffle(queries)
    return queries[:max_queries]


def _collect_candidates(seed: int, max_queries: int) -> list[dict[str, Any]]:
    queries = _build_queries(seed=seed, max_queries=max_queries)
    by_media: dict[str, dict[str, Any]] = {}
    fail = 0
    for i, (tags, mode) in enumerate(queries, start=1):
        try:
            items = _query_feed(tags=tags, tagmode=mode)
        except Exception:
            fail += 1
            if fail <= 8:
                print(f"[warn] feed failed: tags={tags} mode={mode}")
            continue
        for item in items:
            media_url = str((item.get("media") or {}).get("m", ""))
            if not media_url:
                continue
            score = _score_item(item)
            record = dict(item)
            record["score"] = score
            record["query_tags"] = tags
            record["query_mode"] = mode
            prev = by_media.get(media_url)
            if prev is None or float(prev.get("score", 0.0)) < score:
                by_media[media_url] = record
        if i % 40 == 0:
            print(f"[collect] queried={i}/{len(queries)} unique_candidates={len(by_media)}")
        time.sleep(0.06)
    return sorted(by_media.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)


def _write_outputs(out_dir: Path, selected: list[dict[str, Any]], image_size: int) -> None:
    images_dir = out_dir / "images"
    (out_dir / "samples.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "instructions.txt").open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(rec["instruction"] + "\n")

    n = len(selected)
    arr_u8 = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    for i, rec in enumerate(selected):
        with Image.open(images_dir / rec["file_name"]) as im:
            rgb = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr_u8[i] = np.asarray(rgb, dtype=np.uint8)
    arr_f32 = arr_u8.astype(np.float32) / 255.0
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_uint8.npy", arr_u8)
    np.save(out_dir / f"calib_images_{image_size}x{image_size}_float32_01.npy", arr_f32)

    readme = [
        "# Navigation Calibration Data (Indoor / Floor-near)",
        "",
        "- Source: Flickr public feed (`flickr.com/services/feeds/photos_public.gne`)",
        "- Focus: indoor scenes, hallway/corridor/room, floor-near viewpoint terms",
        f"- Samples: {len(selected)}",
        "",
        "## Files",
        "- `images/*.jpg`",
        "- `samples.json`, `samples.jsonl`",
        "- `instructions.txt`",
        f"- `calib_images_{image_size}x{image_size}_uint8.npy`",
        f"- `calib_images_{image_size}x{image_size}_float32_01.npy`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


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

    print("[collect] gathering candidates from Flickr feed...")
    candidates = _collect_candidates(seed=int(args.seed), max_queries=int(args.max_queries))
    print(f"[collect] total candidates={len(candidates)}")

    target_n = int(args.num_samples)
    min_score = float(args.min_score)
    selected: list[dict[str, Any]] = []
    failures = 0
    for idx, item in enumerate(candidates):
        if len(selected) >= target_n:
            break
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue

        media_m = str((item.get("media") or {}).get("m", ""))
        image_urls = _image_candidates_from_media_url(media_m)
        file_name = f"{len(selected):04d}_{abs(hash(media_m)) & 0xffffffff:08x}.jpg"
        out_path = images_dir / file_name
        ok, used_url = _download_image(image_urls, out_path)
        if not ok:
            failures += 1
            continue

        # Validate dimensions and filter too small images.
        try:
            with Image.open(out_path) as im:
                w, h = im.size
            if w < 320 or h < 240:
                out_path.unlink(missing_ok=True)
                continue
        except Exception:
            out_path.unlink(missing_ok=True)
            continue

        rec = {
            "sample_index": len(selected),
            "file_name": file_name,
            "local_image_path": f"images/{file_name}",
            "source_url": used_url,
            "flickr_page": item.get("link"),
            "title": item.get("title"),
            "tags": str(item.get("tags", "")).split(),
            "author": item.get("author"),
            "published": item.get("published"),
            "score": score,
            "query_tags": item.get("query_tags"),
            "query_mode": item.get("query_mode"),
            "instruction": _instruction_from_item(item),
        }
        selected.append(rec)
        if len(selected) % 50 == 0:
            print(f"[progress] downloaded {len(selected)}/{target_n} (failures={failures}, scanned={idx+1})")

    if len(selected) < target_n:
        raise RuntimeError(
            f"Not enough samples: got={len(selected)} target={target_n}. "
            f"Try lowering --min-score or increasing --max-queries."
        )

    _write_outputs(out_dir=out_dir, selected=selected, image_size=int(args.image_size))
    print("[done] calibration dataset generated")
    print(f"[done] output_dir={out_dir}")
    print(f"[done] samples={len(selected)} failures={failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
