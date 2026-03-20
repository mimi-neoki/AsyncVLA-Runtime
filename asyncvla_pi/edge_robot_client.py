from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .image_ring_buffer import ImageRingBuffer
from .pd_controller import PDController
from .policy_payload import build_policy_payload

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass
class EdgeRobotClientConfig:
    policy_url: str
    image_key: str = "front_image"
    timestamp_key: str = "timestamp_ns"
    goal_pose_key: str = "goal_pose"
    current_pose_key: str = "current_pose"
    instruction_key: str = "instruction"
    task_mode_key: str = "task_mode"
    task_id_key: str = "task_id"
    satellite_key: str = "satellite"
    default_instruction: str | None = None
    default_task_mode: str | None = None
    default_task_id: int | None = None
    default_satellite: bool | None = None
    camera_hz: float = 15.0
    edge_hz: float = 8.0
    policy_hz: float = 8.0
    policy_timeout_s: float = 1.0
    ring_capacity: int = 256
    nearest_frame_max_delta_ms: float = 250.0
    jpeg_quality: int = 85
    metric_waypoint_spacing: float = 0.1


@dataclass
class GuidanceCache:
    projected_tokens: np.ndarray
    timestamp_ns: int


class EdgeAwareRobotClient:
    """Pi-side async client using latest guidance cache + delayed frame lookup."""

    def __init__(
        self,
        robot: Any,
        edge_runner: Any,
        pd_controller: PDController,
        config: EdgeRobotClientConfig,
    ) -> None:
        self.robot = robot
        self.edge_runner = edge_runner
        self.pd_controller = pd_controller
        self.config = config

        self.ring_buffer = ImageRingBuffer(capacity=config.ring_capacity)
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest_observation: dict[str, Any] | None = None
        self._latest_guidance: GuidanceCache | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if requests is None:
            raise RuntimeError("requests is required for policy server communication")
        self.robot.connect()
        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._policy_loop, name="policy", daemon=True),
            threading.Thread(target=self._control_loop, name="control", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._running.clear()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.robot.disconnect()

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _extract_image(self, observation: dict[str, Any]) -> np.ndarray:
        image = observation.get(self.config.image_key)
        if image is None and isinstance(observation.get("images"), dict):
            image = observation["images"].get(self.config.image_key)
        if image is None:
            raise KeyError(f"Missing image key: {self.config.image_key}")
        return np.asarray(image)

    def _capture_loop(self) -> None:
        period = 1.0 / max(self.config.camera_hz, 1e-6)
        while self._running.is_set():
            started = time.monotonic()
            obs = self.robot.get_observation()
            image = self._extract_image(obs)
            ts = int(obs.get(self.config.timestamp_key, time.monotonic_ns()))
            self.ring_buffer.push(image, ts, metadata={"observation": obs})
            with self._lock:
                self._latest_observation = obs
            sleep_s = max(0.0, period - (time.monotonic() - started))
            time.sleep(sleep_s)

    def _encode_image(self, image: np.ndarray) -> str:
        if cv2 is not None:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
            ok, buf = cv2.imencode(".jpg", image, params)
            if not ok:
                raise RuntimeError("Failed to encode image")
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return base64.b64encode(np.asarray(image).tobytes()).decode("ascii")

    def _build_policy_payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        image = self._extract_image(observation)
        instruction = observation.get(self.config.instruction_key, self.config.default_instruction)
        task_mode = observation.get(self.config.task_mode_key, self.config.default_task_mode)
        task_id = observation.get(self.config.task_id_key, self.config.default_task_id)
        satellite = observation.get(self.config.satellite_key, self.config.default_satellite)
        return build_policy_payload(
            image=image,
            encoded_image=self._encode_image(image),
            timestamp_ns=int(observation.get(self.config.timestamp_key, time.monotonic_ns())),
            current_pose=observation.get(self.config.current_pose_key, [0, 0, 0]),
            goal_pose=observation.get(self.config.goal_pose_key),
            instruction=instruction,
            task_mode=task_mode,
            task_id=task_id,
            satellite=satellite,
            image_key=self.config.image_key,
            timestamp_key=self.config.timestamp_key,
            current_pose_key=self.config.current_pose_key,
            goal_pose_key=self.config.goal_pose_key,
            instruction_key=self.config.instruction_key,
            task_mode_key=self.config.task_mode_key,
            task_id_key=self.config.task_id_key,
            satellite_key=self.config.satellite_key,
            metric_waypoint_spacing=self.config.metric_waypoint_spacing,
        )

    def _policy_loop(self) -> None:
        period = 1.0 / max(self.config.policy_hz, 1e-6)
        max_delta_ns = int(self.config.nearest_frame_max_delta_ms * 1e6)
        while self._running.is_set():
            started = time.monotonic()
            with self._lock:
                obs = dict(self._latest_observation) if self._latest_observation is not None else None
            if obs is not None:
                try:
                    payload = self._build_policy_payload(obs)
                    resp = requests.post(
                        self.config.policy_url,
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                        timeout=self.config.policy_timeout_s,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    tokens = np.asarray(body["projected_tokens"], dtype=np.float32)
                    timestamp_ns = int(body.get(self.config.timestamp_key, payload[self.config.timestamp_key]))
                    if self.ring_buffer.nearest(timestamp_ns, max_delta_ns=max_delta_ns) is not None:
                        with self._lock:
                            self._latest_guidance = GuidanceCache(tokens, timestamp_ns)
                except Exception:
                    pass
            sleep_s = max(0.0, period - (time.monotonic() - started))
            time.sleep(sleep_s)

    def _control_loop(self) -> None:
        period = 1.0 / max(self.config.edge_hz, 1e-6)
        max_delta_ns = int(self.config.nearest_frame_max_delta_ms * 1e6)
        while self._running.is_set():
            started = time.monotonic()
            with self._lock:
                obs = dict(self._latest_observation) if self._latest_observation is not None else None
                guidance = self._latest_guidance

            if obs is not None and guidance is not None:
                current_image = self._extract_image(obs)
                delayed = self.ring_buffer.nearest(guidance.timestamp_ns, max_delta_ns=max_delta_ns)
                if delayed is not None:
                    goal_pose = np.asarray(obs.get(self.config.goal_pose_key, [0, 0, 0]), dtype=np.float32)
                    raw_out = self.edge_runner.infer(
                        current_image=current_image,
                        delayed_image=delayed.frame,
                        projected_tokens=guidance.projected_tokens,
                        goal_pose=goal_pose,
                    )
                    cmd = self.pd_controller.compute_cmd(
                        pose_chunk=raw_out,
                        metric_waypoint_spacing=self.config.metric_waypoint_spacing,
                    )
                    self.robot.send_action(cmd)

            sleep_s = max(0.0, period - (time.monotonic() - started))
            time.sleep(sleep_s)
