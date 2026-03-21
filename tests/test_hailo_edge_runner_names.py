from asyncvla_pi.hailo_edge_runner import HailoEdgeRunner


def test_resolve_stream_name_exact_match() -> None:
    available = ["edge/input_layer2", "edge/input_layer1", "edge/input_layer3"]
    assert HailoEdgeRunner._resolve_stream_name("edge/input_layer2", available) == "edge/input_layer2"


def test_resolve_stream_name_by_leaf() -> None:
    available = [
        "edge_adapter_static/input_layer2",
        "edge_adapter_static/input_layer1",
        "edge_adapter_static/input_layer3",
    ]
    assert (
        HailoEdgeRunner._resolve_stream_name("edge/input_layer2", available)
        == "edge_adapter_static/input_layer2"
    )


def test_resolve_stream_name_no_match_returns_requested() -> None:
    available = ["edge_adapter_static/input_layer2"]
    assert HailoEdgeRunner._resolve_stream_name("edge/input_layer1", available) == "edge/input_layer1"
