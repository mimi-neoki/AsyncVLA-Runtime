# uv run scripts/demo_edge_server_visual.py \
uv run scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --hef models/edge_adapter_v520.hef \
    --instruction "move to the green tea plastic bottle" \
    --task-mode language_only \
    --no-satellite \
    --show

# ssh -N -L 8000:0.0.0.0:8000 a100-highreso