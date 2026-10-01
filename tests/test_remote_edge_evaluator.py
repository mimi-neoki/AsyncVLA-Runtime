import numpy as np

from scripts.evaluate_remote_edge_hef_candidates import _build_payload, _payload_image_rgb
from scripts.run_pi_edge_compare_server import _decode_image_blob


def test_hybrid_diagnostics_capture_the_feature_used_by_head():
    import hashlib
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    from asyncvla_pi.hailo_edge_runner import HailoEdgeRunner, HailoEdgeRunnerConfig
    from asyncvla_pi.hybrid_edge_runner import HybridEdgeRunner
    from scripts.run_pi_edge_compare_server import EdgeCompareService

    feature = np.arange(1024, dtype=np.float32).reshape(1, 1, 1024)
    hybrid = HybridEdgeRunner.__new__(HybridEdgeRunner)
    hybrid.config = SimpleNamespace(fused_dim=1024)
    hybrid.device = torch.device("cpu")
    hybrid.dtype = torch.float32
    hybrid.hailo_runner = HailoEdgeRunner(
        HailoEdgeRunnerConfig(hef_path="fake.hef", chunk_size=1, pose_dim=1024),
        fallback_fn=lambda _: feature,
    )
    hybrid.model = SimpleNamespace(predict_action_from_fused=lambda x: x[:, :32].reshape(1, 8, 4))
    service = EdgeCompareService.__new__(EdgeCompareService)
    service.lock = threading.Lock()
    service.hef_backend = "hef_torch_head"
    service.hef_runner = hybrid
    service.torch_runner = SimpleNamespace(infer=lambda **_: feature[:, :, :32].reshape(1, 8, 4))
    service.active_hef_path = Path("fake.hef")
    service.allclose_rtol = service.allclose_atol = 0.01
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    payload = _build_payload(image, image, np.zeros((8, 1024)), 90, image_encoding="raw")
    payload["include_diagnostics"] = True
    result = service.infer(payload)
    assert result["diff_report"]["allclose"]
    np.testing.assert_array_equal(result["diagnostics"]["hef_fused_feature"], feature.reshape(1, 1024))
    assert result["diagnostics"]["output_scales"] is None
    prepared = hybrid.hailo_runner._build_inputs(image, image, np.zeros((8, 1024)), None)
    for name, values in prepared.items():
        info = result["diagnostics"]["input_buffers"][name]
        assert info["sha256"] == hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        assert info["dtype"] == str(values.dtype)


def test_local_reference_uses_server_decoded_pixels():
    image = np.random.default_rng(42).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    payload = _build_payload(image, image, np.zeros((8, 1024)), jpeg_quality=70)
    decoded = _payload_image_rgb(payload["current_image"])
    assert not np.array_equal(decoded, image)
    np.testing.assert_array_equal(decoded, _decode_image_blob(payload["current_image"]))


def test_raw_transport_preserves_pixels():
    image = np.random.default_rng(42).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    payload = _build_payload(image, image, np.zeros((8, 1024)), jpeg_quality=90, image_encoding="raw")
    np.testing.assert_array_equal(image, _payload_image_rgb(payload["current_image"]))
    np.testing.assert_array_equal(image, _decode_image_blob(payload["current_image"]))


def test_warmup_does_not_remove_evaluation_samples(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts import evaluate_remote_edge_hef_candidates as evaluator

    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"local_image_path": f"{i}.png"} for i in range(3)]))
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, np.zeros((3, 8, 1024), dtype=np.float32))
    hef = tmp_path / "fake.hef"
    hef.touch()
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        check_health=False, samples_json=str(samples), images_dir=str(tmp_path),
        projected_tokens=str(tokens), hef=[str(hef)], hef_glob="", num_samples=3,
        start_index=0, sample_strategy="first", seed=0, edge_url="http://unused/infer",
        timeout_s=1, compare_reference="server", delayed_mode="same", jpeg_quality=90,
        image_encoding="raw", include_diagnostics=False, warmup=2, save_json=str(output),
    )
    calls = []
    array = np.ones((1, 8, 4), dtype=np.float32)

    def post(*_):
        calls.append(1)
        return {"diff_report": evaluator._compare_outputs(array, array),
                "latency_ms": {"hef": 1, "torch": 1, "total": 2},
                "torch_action_chunk": array.tolist(), "hef_action_chunk": array.tolist()}

    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    monkeypatch.setattr(evaluator, "_upload_hef", lambda *_: {})
    monkeypatch.setattr(evaluator, "_load_rgb_image", lambda *_: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(evaluator, "_post_infer", post)
    assert evaluator.main() == 0
    report = json.loads(output.read_text())
    assert len(calls) == 5
    assert report["candidates"][0]["summary"]["samples"] == 3
    assert [s["sample_index"] for s in report["candidates"][0]["samples"]] == [0, 1, 2]


def test_stage_probe_sends_saved_inputs_and_keeps_all_samples(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import evaluate_edge_stage_probe_remote as evaluator

    rng = np.random.default_rng(1)
    current = rng.integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    delayed = current[::-1].copy()
    tokens = rng.normal(size=(2, 1, 8, 1024)).astype(np.float32)
    action = np.ones((2, 8, 4), dtype=np.float32)
    feature = np.ones((2, 1024), dtype=np.float32)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, indices=[3, 7], runtime_current_image=current,
             runtime_delayed_image=delayed, native_projected_tokens=tokens,
             hf_action=action, hf_fused=feature)
    output = tmp_path / "report.json"
    hef = tmp_path / "fake.hef"
    hef.touch()
    monkeypatch.setattr(sys, "argv", ["probe", "--inputs", str(inputs), "--hef", str(hef),
                                     "--save-json", str(output), "--warmup", "2"])
    monkeypatch.setattr(evaluator, "_upload_hef", lambda *_: {})
    payloads = []

    def post(_url, _timeout, payload):
        payloads.append(payload)
        return {"hef_action_chunk": action[:1].tolist(),
                "torch_action_chunk": action[:1].tolist(),
                "latency_ms": {"hef": 1, "torch": 2, "total": 3},
                "diagnostics": {"hef_fused_feature": feature[:1].tolist()}}

    monkeypatch.setattr(evaluator, "_post_infer", post)
    evaluator.main()
    report = json.loads(output.read_text())
    results = report["candidates"][0]["samples"]
    assert len(payloads) == 4
    assert [s["sample_index"] for s in results] == [3, 7]
    assert all(s["fused_diff"]["rmse"] == 0 for s in results)
    for pos, payload in enumerate(payloads[2:]):
        np.testing.assert_array_equal(_decode_image_blob(payload["current_image"]), current[pos])
        np.testing.assert_array_equal(_decode_image_blob(payload["delayed_image"]), delayed[pos])
        np.testing.assert_array_equal(payload["projected_tokens"], tokens[pos, 0])
        assert payload["include_diagnostics"]
