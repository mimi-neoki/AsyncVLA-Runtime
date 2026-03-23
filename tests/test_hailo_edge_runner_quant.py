import numpy as np

from asyncvla_pi.hailo_edge_runner import HailoEdgeRunner, HailoEdgeRunnerConfig


def test_dequantize_output_scalar_quant_info() -> None:
    runner = HailoEdgeRunner(HailoEdgeRunnerConfig(hef_path="dummy.hef"))
    runner._get_output_quant_info = lambda _name: (  # type: ignore[method-assign]
        np.asarray([0.5], dtype=np.float32),
        np.asarray([10.0], dtype=np.float32),
    )

    raw = np.asarray([[[12, 14], [16, 18]]], dtype=np.uint8)
    out = runner._dequantize_output("out", raw)

    expected = (raw.astype(np.float32) - 10.0) * 0.5
    assert np.allclose(out, expected)


def test_dequantize_output_per_channel_last_dim() -> None:
    runner = HailoEdgeRunner(HailoEdgeRunnerConfig(hef_path="dummy.hef"))
    runner._get_output_quant_info = lambda _name: (  # type: ignore[method-assign]
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    raw = np.asarray([[[[11, 12, 13]]]], dtype=np.uint8)
    out = runner._dequantize_output("out", raw)

    expected = np.asarray([[[[(11 - 1) * 0.1, (12 - 2) * 0.2, (13 - 3) * 0.3]]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_dequantize_output_per_channel_second_dim() -> None:
    runner = HailoEdgeRunner(HailoEdgeRunnerConfig(hef_path="dummy.hef"))
    runner._get_output_quant_info = lambda _name: (  # type: ignore[method-assign]
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    raw = np.asarray([[[11]], [[12]], [[13]]], dtype=np.uint8).reshape(1, 3, 1)
    out = runner._dequantize_output("out", raw)

    expected = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    assert np.allclose(out, expected)
