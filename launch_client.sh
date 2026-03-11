#!/usr/bin/env bash
set -euo pipefail

# OBJECT_CHECK_MODE="${1:-not_found_only}"
OBJECT_CHECK_MODE="${1:-always}"
case "${OBJECT_CHECK_MODE}" in
  not_found_only|always)
    ;;
  *)
    echo "Usage: $0 [not_found_only|always]" >&2
    exit 1
    ;;
esac

# uv run scripts/demo_edge_server_visual.py \
echo "Runtime instruction update is enabled."
echo "After start, type a new target noun phrase and press Enter in this terminal."
echo "Object check mode: ${OBJECT_CHECK_MODE}"
echo "Each stdin update updates target noun phrase for object check."
echo "Mode=not_found_only: periodic check only while NOT FOUND."
echo "Mode=always: periodic check always runs regardless of FOUND/NOT FOUND."
echo "If NOT FOUND, LeKiwi rotates in place until FOUND."

uv run scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --hef models/edge_adapter_v520.hef \
    --instruction-verb "move to" \
    --instruction-noun "bottle" \
    --stdin-object-check \
    --yolo-hef models/yolo_world_v2s.hef \
    --clip-hef models/clip_vit_b_32_text_encoder.hef \
    --yolo-conf-thres 0.50 \
    --stdin-object-check-mode "${OBJECT_CHECK_MODE}" \
    --task-mode language_only \
    --no-satellite \
    --show

# ssh -N -L 8000:0.0.0.0:8000 a100-highreso
