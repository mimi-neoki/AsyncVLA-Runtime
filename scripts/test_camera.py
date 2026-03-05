#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raspi_mobile_robot import RaspiMobileRobot, RaspiMobileRobotConfig

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera test for RaspiMobileRobot")
    parser.add_argument(
        "--camera-index",
        default="auto",
        help="Camera index or device path (e.g. 0, 1, /dev/video0). Use 'auto' to probe.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "opencv", "rpicam"],
        default="auto",
        help="Capture backend. 'auto' tries OpenCV first, then rpicam-jpeg fallback.",
    )
    parser.add_argument("--rpicam-camera", type=int, default=0, help="Camera id for rpicam-jpeg backend")
    parser.add_argument("--rpicam-timeout-ms", type=int, default=150, help="Timeout for each rpicam-jpeg capture")
    parser.add_argument(
        "--libcamerify",
        choices=["auto", "off", "on"],
        default="auto",
        help="Wrap process with libcamerify for OpenCV camera capture on libcamera systems.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to capture")
    parser.add_argument("--interval-sec", type=float, default=0.1, help="Sleep between captures")
    parser.add_argument("--show", action="store_true", help="Show preview window (requires OpenCV GUI)")
    parser.add_argument("--save-dir", default="", help="If set, save captured frames as JPEG")
    parser.add_argument("--prefix", default="frame", help="Filename prefix when --save-dir is set")
    parser.add_argument("--probe-max-index", type=int, default=40, help="Used only when --camera-index auto")
    parser.add_argument(
        "--probe-dev-nodes",
        action="store_true",
        help="Also probe explicit /dev/video* paths in auto mode (can be slow on some drivers)",
    )
    return parser.parse_args()


def _list_video_devices() -> list[str]:
    return [str(p) for p in sorted(Path("/dev").glob("video*"))]


def _can_open_and_read(device: int | str) -> bool:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    ok = bool(cap.isOpened())
    read_ok = False
    if ok:
        read_ok, _ = cap.read()
    cap.release()
    return bool(ok and read_ok)


def _auto_pick_camera(probe_max_index: int, probe_dev_nodes: bool) -> int | str:
    # Probe numeric indices first, this is usually faster on Raspberry Pi.
    for idx in range(max(probe_max_index, 0) + 1):
        if _can_open_and_read(idx):
            return idx
    if probe_dev_nodes:
        for dev in _list_video_devices():
            if _can_open_and_read(dev):
                return dev
    raise RuntimeError("No readable camera device found")


def _print_camera_diagnostics() -> None:
    devices = _list_video_devices()
    print("Detected /dev/video*: " + (", ".join(devices) if devices else "(none)"))
    try:
        proc = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            check=False,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout + proc.stderr).strip()
        if out:
            print("rpicam-hello --list-cameras:")
            print(out)
    except FileNotFoundError:
        pass


def _capture_frame_rpicam(
    camera: int,
    width: int,
    height: int,
    timeout_ms: int,
    out_jpg: Path,
) -> np.ndarray:
    cmd = [
        "rpicam-jpeg",
        "--camera",
        str(camera),
        "--nopreview",
        "--immediate",
        "--timeout",
        str(max(timeout_ms, 1)),
        "--width",
        str(width),
        "--height",
        str(height),
        "--output",
        str(out_jpg),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "rpicam-jpeg failed: "
            f"returncode={proc.returncode}, stderr={proc.stderr.strip() or '(empty)'}"
        )
    frame = cv2.imread(str(out_jpg), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode rpicam output image: {out_jpg}")
    return frame


def _parse_camera_arg(raw: str, probe_max_index: int, probe_dev_nodes: bool) -> int | str:
    if raw.lower() == "auto":
        return _auto_pick_camera(probe_max_index, probe_dev_nodes)
    if raw.startswith("/dev/video"):
        return raw
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Invalid --camera-index: {raw}") from exc


def _reexec_with_libcamerify() -> None:
    libcamerify = shutil.which("libcamerify")
    if not libcamerify:
        raise RuntimeError("libcamerify command not found")
    env = dict(os.environ)
    env["ASYNCVLA_LIBCAMERIFY_ACTIVE"] = "1"
    cmd = [libcamerify, sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    print("Re-exec with libcamerify for OpenCV camera capture...")
    os.execvpe(libcamerify, cmd, env)


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required for camera test (cv2 import failed)")
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    robot: RaspiMobileRobot | None = None
    capture_backend = args.backend
    camera_index: int | str | None = None
    rpicam_out = Path(f"/tmp/asyncvla_cam_test_{os.getpid()}.jpg")
    libcamerify_active = os.environ.get("ASYNCVLA_LIBCAMERIFY_ACTIVE") == "1"

    if capture_backend in {"auto", "opencv"}:
        try:
            camera_index = _parse_camera_arg(
                args.camera_index,
                args.probe_max_index,
                args.probe_dev_nodes,
            )
            robot = RaspiMobileRobot(
                config=RaspiMobileRobotConfig(
                    camera_index=camera_index,  # type: ignore[arg-type]
                    camera_width=args.width,
                    camera_height=args.height,
                    camera_fps=args.fps,
                )
            )
            robot.connect()
            capture_backend = "opencv"
        except Exception as exc:
            if args.libcamerify in {"auto", "on"} and not libcamerify_active:
                _reexec_with_libcamerify()
            if args.backend == "opencv":
                _print_camera_diagnostics()
                raise RuntimeError(f"OpenCV camera selection failed: {exc}") from exc
            # auto mode fallback to rpicam
            capture_backend = "rpicam"

    if capture_backend == "rpicam":
        if not Path("/usr/bin/rpicam-jpeg").exists():
            _print_camera_diagnostics()
            raise RuntimeError("rpicam-jpeg not found, cannot use rpicam backend")

    print(
        f"Camera test start: backend={capture_backend}, "
        f"index={camera_index if capture_backend == 'opencv' else args.rpicam_camera}, "
        f"size={args.width}x{args.height}, fps={args.fps}, num_frames={args.num_frames}"
    )
    captured = 0
    started = time.monotonic()

    try:
        for i in range(args.num_frames):
            if capture_backend == "opencv":
                assert robot is not None
                obs = robot.get_observation()
                frame = np.asarray(obs[robot.config.image_key])
                ts = int(obs[robot.config.timestamp_key])
            else:
                frame = _capture_frame_rpicam(
                    camera=args.rpicam_camera,
                    width=args.width,
                    height=args.height,
                    timeout_ms=args.rpicam_timeout_ms,
                    out_jpg=rpicam_out,
                )
                ts = time.monotonic_ns()
            print(f"[{i:04d}] ts={ts} shape={frame.shape} dtype={frame.dtype}")

            if save_dir is not None:
                out_path = save_dir / f"{args.prefix}_{i:04d}_{ts}.jpg"
                ok = cv2.imwrite(str(out_path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to save frame: {out_path}")

            if args.show:
                cv2.imshow("camera_test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("Stop requested from keyboard.")
                    captured += 1
                    break

            captured += 1
            if args.interval_sec > 0:
                time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if robot is not None:
            robot.disconnect()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started, 1e-6)
    print(f"Captured {captured} frames in {elapsed:.2f}s ({captured / elapsed:.2f} fps)")
    if save_dir is not None:
        print(f"Saved frames to: {save_dir}")


if __name__ == "__main__":
    main()
