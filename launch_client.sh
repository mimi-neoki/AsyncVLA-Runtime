#!/usr/bin/env bash
set -euo pipefail

# uv run scripts/demo_edge_server_visual.py \
echo "Runtime instruction update is enabled."
echo "After start, type a new target noun phrase and press Enter in this terminal."
echo "Each stdin update triggers one YOLO-World object-presence check on the latest camera frame."

uv run scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --hef models/edge_adapter_v520.hef \
    --instruction-verb "move to" \
    --instruction-noun "the orange pet bottle" \
    --stdin-object-check \
    --yolo-hef models/yolo_world_v2s.hef \
    --clip-hef models/clip_vit_b_32_text_encoder.hef \
    --yolo-conf-thres 0.50 \
    --task-mode language_only \
    --no-satellite \
    --show

# ssh -N -L 8000:0.0.0.0:8000 a100-highreso
