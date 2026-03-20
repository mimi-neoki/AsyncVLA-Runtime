from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pi_edge_client.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_run_pi_edge_client_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyRobot:
    def __init__(self, config, odom_provider, goal_pose_provider, cmd_vel_publisher) -> None:
        self.config = config
        self.odom_provider = odom_provider
        self.goal_pose_provider = goal_pose_provider
        self.cmd_vel_publisher = cmd_vel_publisher


class _DummyPDController:
    def __init__(self, config) -> None:
        self.config = config


class _DummyClient:
    def __init__(self, robot, edge_runner, pd_controller, config) -> None:
        self.robot = robot
        self.edge_runner = edge_runner
        self.pd_controller = pd_controller
        self.config = config
        self.ran = False

    def run_forever(self) -> None:
        self.ran = True


def test_main_uses_torch_edge_runner_for_hf_backend(monkeypatch, tmp_path) -> None:
    module = _load_module()
    calls: dict[str, object] = {}

    class _DummyTorchRunner:
        def __init__(self, config) -> None:
            calls["torch_runner_config"] = config

    def _fail_hailo_runtime() -> None:
        raise AssertionError("Hailo runtime should not be required for hf backend")

    def _fail_hailo_runner(*args, **kwargs):
        raise AssertionError("Hailo edge runner should not be constructed for hf backend")

    monkeypatch.setattr(module, "TorchEdgeRunner", _DummyTorchRunner)
    monkeypatch.setattr(module, "HailoEdgeRunner", _fail_hailo_runner)
    monkeypatch.setattr(module, "_ensure_hailo_runtime_available", _fail_hailo_runtime)
    monkeypatch.setattr(module, "RaspiMobileRobot", _DummyRobot)
    monkeypatch.setattr(module, "PDController", _DummyPDController)
    monkeypatch.setattr(module, "EdgeAwareRobotClient", _DummyClient)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--policy-url",
            "http://127.0.0.1:8000/infer",
            "--edge-backend",
            "hf",
            "--hf-dir",
            str(tmp_path),
            "--edge-device",
            "cpu",
            "--edge-dtype",
            "float32",
            "--libcamerify",
            "off",
        ],
    )

    module.main()

    runner_config = calls["torch_runner_config"]
    assert runner_config.hf_dir == str(tmp_path.resolve())
    assert runner_config.device == "cpu"
    assert runner_config.dtype == "float32"


def test_main_uses_hailo_runner_for_hef_backend(monkeypatch, tmp_path) -> None:
    module = _load_module()
    calls: dict[str, object] = {"hailo_runtime_checked": False}
    hef_path = tmp_path / "edge.hef"
    hef_path.write_bytes(b"hef")

    class _DummyHailoRunner:
        def __init__(self, config) -> None:
            calls["hailo_runner_config"] = config

    def _mark_hailo_runtime() -> None:
        calls["hailo_runtime_checked"] = True

    def _fail_torch_runner(*args, **kwargs):
        raise AssertionError("Torch edge runner should not be constructed for hef backend")

    monkeypatch.setattr(module, "HailoEdgeRunner", _DummyHailoRunner)
    monkeypatch.setattr(module, "TorchEdgeRunner", _fail_torch_runner)
    monkeypatch.setattr(module, "_ensure_hailo_runtime_available", _mark_hailo_runtime)
    monkeypatch.setattr(module, "RaspiMobileRobot", _DummyRobot)
    monkeypatch.setattr(module, "PDController", _DummyPDController)
    monkeypatch.setattr(module, "EdgeAwareRobotClient", _DummyClient)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--policy-url",
            "http://127.0.0.1:8000/infer",
            "--edge-backend",
            "hef",
            "--hef",
            str(hef_path),
            "--libcamerify",
            "off",
        ],
    )

    module.main()

    assert calls["hailo_runtime_checked"] is True
    runner_config = calls["hailo_runner_config"]
    assert runner_config.hef_path == str(hef_path.resolve())
