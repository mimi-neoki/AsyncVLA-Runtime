#!/usr/bin/env bash
set -euo pipefail

# uv run scripts/demo_edge_server_visual.py \
echo "Runtime instruction update is enabled."
echo "After start, type a new target noun phrase and press Enter in this terminal."

uv run scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --hef models/edge_adapter_v520.hef \
    --instruction-verb "move to" \
    --instruction-noun "the orange pet bottle" \
    --task-mode language_only \
    --no-satellite \
    --show

# ssh -N -L 8000:0.0.0.0:8000 a100-highreso
