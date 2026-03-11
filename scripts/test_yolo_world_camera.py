#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raspi_mobile_robot import RaspiMobileRobot, RaspiMobileRobotConfig
from test_yolo_world import (
    _build_text_embeddings_with_clip,
    _dequantize,
    _find_inputs,
    _parse_text_classes,
    _prepare_text_embeddings,
)


@dataclass
class YoloHeadSpec:
    reg_name: str
    cls_name: str
    h: int
    w: int
    stride_x: float
    stride_y: float
    reg_scale: float
    reg_zp: float
    cls_scale: float
    cls_zp: float


@dataclass
class YoloWorldHailoRunnerConfig:
    hef_path: str
    input_image_name: str | None = None
    input_text_name: str | None = None
    timeout_ms: int = 10000


class YoloWorldHailoRunner:
    def __init__(self, target: Any, config: YoloWorldHailoRunnerConfig) -> None:
        self.target = target
        self.config = config
        self.infer_model = self.target.create_infer_model(str(config.hef_path))

        self.image_input_name, self.text_input_name = self._resolve_input_names()
        self.input_h, self.input_w, _ = tuple(self.infer_model.input(self.image_input_name).shape)
        self.text_shape = tuple(self.infer_model.input(self.text_input_name).shape)
        self.head_specs = _collect_yolo_heads(self.infer_model, input_h=self.input_h, input_w=self.input_w)

        self._configured_ctx: Any = None
        self._configured: Any = None
        self._bindings: Any = None
        self._text_embeddings: np.ndarray | None = None

    def _resolve_input_names(self) -> tuple[str, str]:
        if self.config.input_image_name and self.config.input_text_name:
            if self.config.input_image_name not in self.infer_model.input_names:
                raise ValueError(
                    f"input_image_name not found: {self.config.input_image_name} "
                    f"(available={self.infer_model.input_names})"
                )
            if self.config.input_text_name not in self.infer_model.input_names:
                raise ValueError(
                    f"input_text_name not found: {self.config.input_text_name} "
                    f"(available={self.infer_model.input_names})"
                )
            return self.config.input_image_name, self.config.input_text_name

        if self.config.input_image_name or self.config.input_text_name:
            raise ValueError("Specify both --input-image-name and --input-text-name, or neither.")
        return _find_inputs(self.infer_model)

    def default_class_names(self) -> list[str]:
        return [f"class_{i}" for i in range(self.text_shape[1])]

    def set_text_embeddings(self, text_emb: np.ndarray) -> None:
        arr = np.asarray(text_emb, dtype=np.uint8)
        if arr.shape != self.text_shape:
            raise ValueError(f"text embedding shape mismatch: expected={self.text_shape}, got={arr.shape}")
        self._text_embeddings = arr
        if self._bindings is not None:
            self._bindings.input(self.text_input_name).set_buffer(self._text_embeddings)

    def __enter__(self) -> YoloWorldHailoRunner:
        self._configured_ctx = self.infer_model.configure()
        self._configured = self._configured_ctx.__enter__()
        self._bindings = self._configured.create_bindings()

        for out_name in self.infer_model.output_names:
            out_shape = tuple(self.infer_model.output(out_name).shape)
            self._bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np.uint16))
        if self._text_embeddings is not None:
            self._bindings.input(self.text_input_name).set_buffer(self._text_embeddings)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._configured_ctx is not None:
            self._configured_ctx.__exit__(exc_type, exc, tb)
        self._configured_ctx = None
        self._configured = None
        self._bindings = None
        return False

    def _set_image_input(self, frame_bgr: np.ndarray) -> None:
        if self._bindings is None:
            raise RuntimeError("Runner is not configured. Use 'with runner:' context.")
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        input_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        self._bindings.input(self.image_input_name).set_buffer(input_rgb)

    def _run_once(self) -> float:
        if self._configured is None or self._bindings is None:
            raise RuntimeError("Runner is not configured. Use 'with runner:' context.")
        if self._text_embeddings is None:
            raise RuntimeError("Text embeddings are not set. Call set_text_embeddings() before inference.")
        t0 = time.perf_counter()
        self._configured.run([self._bindings], self.config.timeout_ms)
        return (time.perf_counter() - t0) * 1000.0

    def warmup(self, frame_bgr: np.ndarray, runs: int = 1) -> None:
        for _ in range(max(0, runs)):
            self._set_image_input(frame_bgr)
            self._run_once()

    def infer(
        self,
        frame_bgr: np.ndarray,
        num_classes: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        self._set_image_input(frame_bgr)
        infer_ms = self._run_once()
        boxes, scores, class_ids = _decode_detections(
            bindings=self._bindings,
            head_specs=self.head_specs,
            num_classes=num_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            input_h=self.input_h,
            input_w=self.input_w,
        )
        return boxes, scores, class_ids, infer_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera demo: CLIP text -> YOLO-World (Hailo) -> OpenCV detection window"
    )
    parser.add_argument("--hef", default="models/yolo_world_v2s.hef", help="Path to YOLO-World HEF")
    parser.add_argument("--clip-hef", default="models/clip_vit_b_32_text_encoder.hef", help="Path to CLIP text HEF")
    parser.add_argument(
        "--text",
        default="person,car,dog",
        help="Comma-separated class texts, e.g. 'person,car,dog'",
    )
    parser.add_argument(
        "--text-embeddings",
        default=None,
        help="Optional .npy text embedding path (if set, --text/--clip-hef path is skipped)",
    )
    parser.add_argument(
        "--clip-model-id",
        default="openai/clip-vit-base-patch32",
        help="HF model id for CLIP tokenizer/embedding table (local_files_only=True)",
    )
    parser.add_argument("--prompt-template", default="{}", help="Prompt template, e.g. 'a photo of {}'")

    parser.add_argument(
        "--camera-index",
        default="0",
        help="Camera index or /dev/video path",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    parser.add_argument(
        "--libcamerify",
        choices=["auto", "off", "on"],
        default="auto",
        help="Wrap process with libcamerify for OpenCV camera capture on libcamera systems.",
    )

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--input-image-name", default=None, help="Optional explicit YOLO image input name")
    parser.add_argument("--input-text-name", default=None, help="Optional explicit YOLO text input name")
    parser.add_argument("--conf-thres", type=float, default=0.10)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means infinite")
    parser.add_argument("--window-name", default="yolo_world_camera")
    parser.add_argument("--save-video", default="")
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser.parse_args()


def _ensure_hailo_runtime_available() -> None:
    try:
        import hailo_platform  # noqa: F401
        return
    except Exception as exc:
        already_fallback = os.environ.get("ASYNCVLA_SYSTEMPY_ACTIVE") == "1"
        system_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else shutil.which("python3")
        current_python = Path(sys.executable).resolve()
        if (
            not already_fallback
            and system_python
            and Path(system_python).resolve() != current_python
        ):
            env = dict(os.environ)
            env["ASYNCVLA_SYSTEMPY_ACTIVE"] = "1"
            cmd = [system_python, str(Path(__file__).resolve()), *sys.argv[1:]]
            print(
                "Hailo runtime import failed in current interpreter; "
                "retrying with system python3..."
            )
            os.execvpe(system_python, cmd, env)
        raise RuntimeError(
            "pyhailort is not available in this Python environment. "
            "Install matching hailort package for this interpreter or run with system python3."
        ) from exc


def _reexec_with_libcamerify() -> None:
    libcamerify = shutil.which("libcamerify")
    if not libcamerify:
        raise RuntimeError("libcamerify command not found")
    env = dict(os.environ)
    env["ASYNCVLA_LIBCAMERIFY_ACTIVE"] = "1"
    cmd = [libcamerify, sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    print("Re-exec with libcamerify for OpenCV camera capture...")
    os.execvpe(libcamerify, cmd, env)


def _parse_camera_index(raw: str) -> int | str:
    if raw.startswith("/dev/video"):
        return raw
    return int(raw)


def _collect_yolo_heads(infer_model, input_h: int, input_w: int) -> list[YoloHeadSpec]:
    reg_by_shape: dict[tuple[int, int], str] = {}
    cls_by_shape: dict[tuple[int, int], str] = {}

    for name in infer_model.output_names:
        shape = tuple(infer_model.output(name).shape)
        if len(shape) != 3:
            continue
        h, w, c = shape
        if c == 64:
            reg_by_shape[(h, w)] = name
        elif c >= 80:
            cls_by_shape[(h, w)] = name

    shared_shapes = sorted(set(reg_by_shape) & set(cls_by_shape), key=lambda hw: hw[0], reverse=True)
    if not shared_shapes:
        raise RuntimeError(
            "Could not find YOLO head output pairs (reg=64ch, cls>=80ch). "
            f"Output names: {infer_model.output_names}"
        )

    specs: list[YoloHeadSpec] = []
    for h, w in shared_shapes:
        reg_name = reg_by_shape[(h, w)]
        cls_name = cls_by_shape[(h, w)]
        reg_q = infer_model.output(reg_name).quant_infos[0]
        cls_q = infer_model.output(cls_name).quant_infos[0]
        specs.append(
            YoloHeadSpec(
                reg_name=reg_name,
                cls_name=cls_name,
                h=h,
                w=w,
                stride_x=float(input_w) / float(w),
                stride_y=float(input_h) / float(h),
                reg_scale=float(reg_q.qp_scale),
                reg_zp=float(reg_q.qp_zp),
                cls_scale=float(cls_q.qp_scale),
                cls_zp=float(cls_q.qp_zp),
            )
        )
    return specs


def _bbox_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-6)
    return inter / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_keep: int) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_keep:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _bbox_iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thres]
    return np.asarray(keep, dtype=np.int32)


def _decode_detections(
    bindings,
    head_specs: list[YoloHeadSpec],
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    input_h: int,
    input_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.arange(16, dtype=np.float32)[None, None, :]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []

    for spec in head_specs:
        reg_q = bindings.output(spec.reg_name).get_buffer()
        cls_q = bindings.output(spec.cls_name).get_buffer()

        reg = _dequantize(reg_q, spec.reg_scale, spec.reg_zp)
        cls = _dequantize(cls_q, spec.cls_scale, spec.cls_zp)
        cls = cls[..., :num_classes]
        if cls.shape[-1] == 0:
            continue

        if float(cls.max()) > 1.0 or float(cls.min()) < 0.0:
            cls = 1.0 / (1.0 + np.exp(-np.clip(cls, -30.0, 30.0)))
        else:
            cls = np.clip(cls, 0.0, 1.0)

        reg = reg.reshape(-1, 4, 16)
        reg = reg - reg.max(axis=2, keepdims=True)
        reg_exp = np.exp(reg)
        reg_prob = reg_exp / (reg_exp.sum(axis=2, keepdims=True) + 1e-9)
        dist = (reg_prob * bins).sum(axis=2)

        cls_flat = cls.reshape(-1, cls.shape[-1])
        scores = cls_flat.max(axis=1)
        class_ids = cls_flat.argmax(axis=1)
        keep = scores >= conf_thres
        if not np.any(keep):
            continue

        idxs = np.where(keep)[0]
        scores = scores[keep]
        class_ids = class_ids[keep]
        dist = dist[keep]

        ys = idxs // spec.w
        xs = idxs % spec.w
        cx = (xs.astype(np.float32) + 0.5) * spec.stride_x
        cy = (ys.astype(np.float32) + 0.5) * spec.stride_y

        l = dist[:, 0] * spec.stride_x
        t = dist[:, 1] * spec.stride_y
        r = dist[:, 2] * spec.stride_x
        b = dist[:, 3] * spec.stride_y

        x1 = np.clip(cx - l, 0.0, float(input_w - 1))
        y1 = np.clip(cy - t, 0.0, float(input_h - 1))
        x2 = np.clip(cx + r, 0.0, float(input_w - 1))
        y2 = np.clip(cy + b, 0.0, float(input_h - 1))

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores.astype(np.float32))
        all_class_ids.append(class_ids.astype(np.int32))

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    keep_global: list[int] = []
    for class_id in np.unique(class_ids):
        idxs = np.where(class_ids == class_id)[0]
        keep_local = _nms(boxes[idxs], scores[idxs], iou_thres=iou_thres, max_keep=max_det)
        keep_global.extend(idxs[keep_local].tolist())

    if not keep_global:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    keep_arr = np.asarray(keep_global, dtype=np.int32)
    order = np.argsort(scores[keep_arr])[::-1]
    keep_arr = keep_arr[order[:max_det]]
    return boxes[keep_arr], scores[keep_arr], class_ids[keep_arr]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return (
        int((37 * class_id + 83) % 255),
        int((17 * class_id + 121) % 255),
        int((29 * class_id + 199) % 255),
    )


def _draw_detections(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    infer_ms: float,
    fps: float,
) -> np.ndarray:
    vis = frame_bgr.copy()
    for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = _color_for_class(int(class_id))
        label = class_names[class_id] if 0 <= int(class_id) < len(class_names) else f"class_{int(class_id)}"
        text = f"{label} {float(score):.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"infer={infer_ms:.1f}ms fps={fps:.2f} det={len(boxes_xyxy)}"
    cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def main() -> int:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required")
    if args.text and args.text_embeddings:
        raise ValueError("Use either --text or --text-embeddings, not both.")

    _ensure_hailo_runtime_available()

    libcamerify_active = os.environ.get("ASYNCVLA_LIBCAMERIFY_ACTIVE") == "1"
    if args.libcamerify == "on" and not libcamerify_active:
        _reexec_with_libcamerify()
    if args.libcamerify == "auto" and not libcamerify_active and shutil.which("libcamerify"):
        _reexec_with_libcamerify()

    hef_path = Path(args.hef).expanduser().resolve()
    if not hef_path.exists():
        raise FileNotFoundError(f"YOLO HEF not found: {hef_path}")

    from hailo_platform import VDevice

    robot = RaspiMobileRobot(
        config=RaspiMobileRobotConfig(
            camera_index=_parse_camera_index(args.camera_index),  # type: ignore[arg-type]
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        ),
        odom_provider=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        goal_pose_provider=lambda: np.array([args.goal_x, args.goal_y, args.goal_yaw], dtype=np.float32),
    )

    writer: cv2.VideoWriter | None = None
    if args.save_video:
        out_path = Path(args.save_video).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, max(args.camera_fps, 1.0), (args.camera_width, args.camera_height))
        print(f"save_video={out_path}")

    print("Camera demo start")
    print(
        f"camera={args.camera_index} backend=RaspiMobileRobot "
        f"size={args.camera_width}x{args.camera_height} fps={args.camera_fps}"
    )
    print(f"Press 'q' or ESC to stop")
    robot.connect()

    try:
        with VDevice() as target:
            yolo_runner = YoloWorldHailoRunner(
                target=target,
                config=YoloWorldHailoRunnerConfig(
                    hef_path=str(hef_path),
                    input_image_name=args.input_image_name,
                    input_text_name=args.input_text_name,
                    timeout_ms=args.timeout_ms,
                ),
            )

            class_names = yolo_runner.default_class_names()
            if args.text_embeddings:
                text_emb = _prepare_text_embeddings(args.text_embeddings, yolo_runner.text_shape, random_text=False)
            elif args.text:
                clip_hef_path = Path(args.clip_hef).expanduser().resolve()
                if not clip_hef_path.exists():
                    raise FileNotFoundError(f"CLIP HEF not found: {clip_hef_path}")
                class_names = _parse_text_classes(args.text)
                text_emb = _build_text_embeddings_with_clip(
                    target=target,
                    clip_hef_path=clip_hef_path,
                    yolo_infer_model=yolo_runner.infer_model,
                    yolo_text_input_name=yolo_runner.text_input_name,
                    class_texts=class_names,
                    clip_model_id=args.clip_model_id,
                    prompt_template=args.prompt_template,
                    timeout_ms=args.timeout_ms,
                )
                print(f"text_classes({len(class_names)}): {class_names}")
            else:
                text_emb = _prepare_text_embeddings(None, yolo_runner.text_shape, random_text=False)
            yolo_runner.set_text_embeddings(text_emb)
            print(
                f"yolo_inputs: image={yolo_runner.image_input_name} text={yolo_runner.text_input_name} "
                f"input={yolo_runner.input_w}x{yolo_runner.input_h}"
            )

            with yolo_runner:

                warmup_done = 0
                frame_count = 0
                started = time.perf_counter()
                read_fail_count = 0

                if args.show:
                    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(args.window_name, args.camera_width, args.camera_height)

                while True:
                    loop_t0 = time.perf_counter()
                    try:
                        obs = robot.get_observation()
                        frame_bgr = np.asarray(obs[robot.config.image_key])
                        ok = frame_bgr is not None and frame_bgr.size > 0
                    except Exception as exc:
                        ok = False
                        frame_bgr = None
                        if read_fail_count in (0, 9, 29):
                            print(f"[WARN] camera read failed ({read_fail_count + 1}): {exc}")
                    if not ok or frame_bgr is None or frame_bgr.size == 0:
                        read_fail_count += 1
                        if args.show:
                            status = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                            cv2.putText(
                                status,
                                "Waiting for camera frame...",
                                (16, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow(args.window_name, status)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                break
                        continue
                    read_fail_count = 0

                    if warmup_done < args.warmup:
                        yolo_runner.warmup(frame_bgr, runs=1)
                        warmup_done += 1
                        if args.show:
                            warmup_vis = frame_bgr.copy()
                            cv2.putText(
                                warmup_vis,
                                f"Warmup {warmup_done}/{args.warmup}",
                                (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow(args.window_name, warmup_vis)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                break
                        continue

                    boxes, scores, class_ids, infer_ms = yolo_runner.infer(
                        frame_bgr=frame_bgr,
                        num_classes=len(class_names),
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det,
                    )

                    if boxes.shape[0] > 0:
                        boxes = boxes.copy()
                        sx = frame_bgr.shape[1] / float(yolo_runner.input_w)
                        sy = frame_bgr.shape[0] / float(yolo_runner.input_h)
                        boxes[:, [0, 2]] *= sx
                        boxes[:, [1, 3]] *= sy

                    loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                    fps = 1000.0 / max(loop_ms, 1e-6)
                    vis = _draw_detections(
                        frame_bgr=frame_bgr,
                        boxes_xyxy=boxes,
                        scores=scores,
                        class_ids=class_ids,
                        class_names=class_names,
                        infer_ms=infer_ms,
                        fps=fps,
                    )

                    if writer is not None:
                        writer.write(vis)
                    if args.show:
                        cv2.imshow(args.window_name, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break

                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = max(time.perf_counter() - started, 1e-6)
                        print(
                            f"frames={frame_count} avg_fps={frame_count / elapsed:.2f} "
                            f"infer_ms={infer_ms:.2f} det={len(boxes)}"
                        )
                    if args.max_frames > 0 and frame_count >= args.max_frames:
                        break
    finally:
        robot.disconnect()
        if writer is not None:
            writer.release()
        if args.show and cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
