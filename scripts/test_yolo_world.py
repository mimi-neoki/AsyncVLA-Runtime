#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal YOLO-World v2s HEF inference on Raspberry Pi 5 + Hailo-10H"
    )
    parser.add_argument("--hef", default="models/yolo_world_v2s.hef", help="Path to HEF file")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, random image is used.")
    parser.add_argument(
        "--text-embeddings",
        default=None,
        help="Optional .npy path for text embeddings (must match model input shape)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Comma-separated class texts. If set, CLIP text encoder HEF is used to build YOLO-World text embeddings.",
    )
    parser.add_argument(
        "--clip-hef",
        default="models/clip_vit_b_32_text_encoder.hef",
        help="Path to CLIP text encoder HEF",
    )
    parser.add_argument(
        "--clip-model-id",
        default="openai/clip-vit-base-patch32",
        help="HF model id for CLIP tokenizer/embedding table (loaded with local_files_only=True)",
    )
    parser.add_argument(
        "--prompt-template",
        default="{}",
        help="Prompt template for each class text, e.g. 'a photo of {}'",
    )
    parser.add_argument(
        "--random-text",
        action="store_true",
        help="Use random uint8 text embeddings instead of zeros when --text-embeddings is not set",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup inference iterations")
    parser.add_argument("--runs", type=int, default=50, help="Measured inference iterations")
    parser.add_argument("--timeout-ms", type=int, default=10000, help="Per-inference timeout in ms")
    return parser.parse_args()


def _find_inputs(infer_model) -> tuple[str, str]:
    image_name = None
    text_name = None
    for name in infer_model.input_names:
        shape = tuple(infer_model.input(name).shape)
        if len(shape) == 3 and shape[-1] == 3:
            image_name = name
        elif len(shape) == 3 and shape[-1] != 3:
            text_name = name
    if image_name is None or text_name is None:
        raise RuntimeError(
            f"Could not auto-detect input names from {infer_model.input_names}. "
            "Please inspect HEF input shapes."
        )
    return image_name, text_name


def _prepare_image(image_path: str | None, expected_shape: tuple[int, ...]) -> np.ndarray:
    h, w, c = expected_shape
    if c != 3:
        raise ValueError(f"Image input must have channel=3, got shape={expected_shape}")

    if image_path is None:
        return np.random.randint(0, 256, size=expected_shape, dtype=np.uint8)

    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-python or omit --image.")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


def _prepare_text_embeddings(
    npy_path: str | None, expected_shape: tuple[int, ...], random_text: bool
) -> np.ndarray:
    if npy_path is not None:
        emb = np.load(npy_path)
        if emb.shape != expected_shape:
            raise ValueError(f"Text embedding shape mismatch: expected={expected_shape}, got={emb.shape}")
        return emb.astype(np.uint8, copy=False)
    if random_text:
        return np.random.randint(0, 256, size=expected_shape, dtype=np.uint8)
    return np.zeros(expected_shape, dtype=np.uint8)


def _quantize(values: np.ndarray, scale: float, zp: float, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    q = np.round(values / float(scale) + float(zp))
    return np.clip(q, info.min, info.max).astype(dtype)


def _dequantize(values: np.ndarray, scale: float, zp: float) -> np.ndarray:
    return (values.astype(np.float32) - float(zp)) * float(scale)


def _parse_text_classes(text_arg: str) -> list[str]:
    texts = [token.strip() for token in text_arg.split(",")]
    texts = [token for token in texts if token]
    if not texts:
        raise ValueError("--text is empty. Pass class names like 'person,car,dog'.")
    return texts


def _build_text_embeddings_with_clip(
    target,
    clip_hef_path: Path,
    yolo_infer_model,
    yolo_text_input_name: str,
    class_texts: list[str],
    clip_model_id: str,
    prompt_template: str,
    timeout_ms: int,
) -> np.ndarray:
    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    except Exception as exc:
        raise RuntimeError(
            "transformers/torch import failed. Install project dependencies in this environment."
        ) from exc

    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_id, local_files_only=True)
    clip_model = CLIPTextModelWithProjection.from_pretrained(clip_model_id, local_files_only=True)
    clip_model.eval()

    prompts = [prompt_template.format(text) for text in class_texts]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]

    with torch.inference_mode():
        clip_input_float = clip_model.text_model.embeddings(input_ids=input_ids).cpu().numpy().astype(np.float32)
    text_projection = clip_model.text_projection.weight.detach().cpu().numpy().astype(np.float32).T

    clip_infer_model = target.create_infer_model(str(clip_hef_path))
    clip_input_name = clip_infer_model.input_names[0]
    clip_output_name = clip_infer_model.output_names[0]
    clip_input_shape = tuple(clip_infer_model.input(clip_input_name).shape)
    clip_output_shape = tuple(clip_infer_model.output(clip_output_name).shape)
    if clip_input_shape != (1, 77, 512):
        raise RuntimeError(f"Unexpected CLIP input shape: {clip_input_shape}")
    if clip_output_shape != (1, 77, 512):
        raise RuntimeError(f"Unexpected CLIP output shape: {clip_output_shape}")

    clip_in_quant = clip_infer_model.input(clip_input_name).quant_infos[0]
    clip_out_quant = clip_infer_model.output(clip_output_name).quant_infos[0]

    yolo_text_shape = tuple(yolo_infer_model.input(yolo_text_input_name).shape)
    if len(yolo_text_shape) != 3 or yolo_text_shape[0] != 1:
        raise RuntimeError(f"Unexpected YOLO text input shape: {yolo_text_shape}")
    max_classes = yolo_text_shape[1]
    if len(class_texts) > max_classes:
        raise ValueError(f"Too many classes: {len(class_texts)} > {max_classes}")

    yolo_text_quant = yolo_infer_model.input(yolo_text_input_name).quant_infos[0]
    class_embeddings_float = np.zeros(yolo_text_shape, dtype=np.float32)

    with clip_infer_model.configure() as clip_configured:
        for class_idx in range(len(class_texts)):
            clip_bindings = clip_configured.create_bindings()
            clip_input_u16 = _quantize(
                clip_input_float[class_idx : class_idx + 1],
                clip_in_quant.qp_scale,
                clip_in_quant.qp_zp,
                np.uint16,
            )
            clip_bindings.input(clip_input_name).set_buffer(clip_input_u16)
            clip_out_u8 = np.empty(clip_output_shape, dtype=np.uint8)
            clip_bindings.output(clip_output_name).set_buffer(clip_out_u8)
            clip_configured.run([clip_bindings], timeout_ms)

            clip_seq_u8 = clip_bindings.output(clip_output_name).get_buffer()
            clip_seq_f32 = _dequantize(clip_seq_u8, clip_out_quant.qp_scale, clip_out_quant.qp_zp)

            token_ids = input_ids[class_idx].cpu().numpy()
            eos_positions = np.where(token_ids == tokenizer.eos_token_id)[0]
            eos_idx = int(eos_positions[-1]) if eos_positions.size else int(np.argmax(token_ids))

            pooled = clip_seq_f32[0, eos_idx, :]
            projected = pooled @ text_projection
            projected = projected / (np.linalg.norm(projected) + 1e-6)
            class_embeddings_float[0, class_idx, :] = projected

    return _quantize(
        class_embeddings_float,
        yolo_text_quant.qp_scale,
        yolo_text_quant.qp_zp,
        np.uint8,
    )


def main() -> int:
    args = parse_args()
    if args.text is not None and args.text_embeddings is not None:
        raise ValueError("Use either --text or --text-embeddings, not both.")

    hef_path = Path(args.hef).expanduser().resolve()
    if not hef_path.exists():
        raise FileNotFoundError(f"HEF not found: {hef_path}")

    try:
        from hailo_platform import VDevice
    except Exception as exc:
        raise RuntimeError(
            "hailo_platform import failed. Use the python env where pyhailort is installed."
        ) from exc

    with VDevice() as target:
        infer_model = target.create_infer_model(str(hef_path))
        image_input_name, text_input_name = _find_inputs(infer_model)

        image_shape = tuple(infer_model.input(image_input_name).shape)
        text_shape = tuple(infer_model.input(text_input_name).shape)

        image = _prepare_image(args.image, image_shape)
        if args.text is not None:
            clip_hef_path = Path(args.clip_hef).expanduser().resolve()
            if not clip_hef_path.exists():
                raise FileNotFoundError(f"CLIP HEF not found: {clip_hef_path}")
            class_texts = _parse_text_classes(args.text)
            text_emb = _build_text_embeddings_with_clip(
                target=target,
                clip_hef_path=clip_hef_path,
                yolo_infer_model=infer_model,
                yolo_text_input_name=text_input_name,
                class_texts=class_texts,
                clip_model_id=args.clip_model_id,
                prompt_template=args.prompt_template,
                timeout_ms=args.timeout_ms,
            )
            print(f"text_classes({len(class_texts)}): {class_texts}")
            print(f"clip_hef: {clip_hef_path}")
            print(f"clip_model_id: {args.clip_model_id}")
        else:
            text_emb = _prepare_text_embeddings(args.text_embeddings, text_shape, args.random_text)

        with infer_model.configure() as configured:
            bindings = configured.create_bindings()
            bindings.input(image_input_name).set_buffer(image)
            bindings.input(text_input_name).set_buffer(text_emb)

            for out_name in infer_model.output_names:
                out_shape = tuple(infer_model.output(out_name).shape)
                out_buf = np.empty(out_shape, dtype=np.uint16)
                bindings.output(out_name).set_buffer(out_buf)

            for _ in range(args.warmup):
                configured.run([bindings], args.timeout_ms)

            latencies_ms: list[float] = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                configured.run([bindings], args.timeout_ms)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            print(f"HEF: {hef_path}")
            print(f"image_input: {image_input_name} shape={image.shape} dtype={image.dtype}")
            print(f"text_input:  {text_input_name} shape={text_emb.shape} dtype={text_emb.dtype}")
            print(
                f"benchmark: warmup={args.warmup} runs={args.runs} timeout_ms={args.timeout_ms}"
            )
            if latencies_ms:
                lat = np.asarray(latencies_ms, dtype=np.float64)
                mean_ms = float(lat.mean())
                fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
                print(
                    "latency_ms: "
                    f"mean={mean_ms:.3f} std={float(lat.std()):.3f} "
                    f"min={float(lat.min()):.3f} p50={float(np.percentile(lat, 50)):.3f} "
                    f"p90={float(np.percentile(lat, 90)):.3f} p95={float(np.percentile(lat, 95)):.3f} "
                    f"max={float(lat.max()):.3f}"
                )
                print(f"throughput_fps_estimate: {fps:.2f}")
            print("outputs:")
            for out_name in infer_model.output_names:
                out = bindings.output(out_name).get_buffer()
                print(
                    f"  {out_name}: shape={out.shape} dtype={out.dtype} "
                    f"min={int(out.min())} max={int(out.max())} mean={float(out.mean()):.2f}"
                )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
