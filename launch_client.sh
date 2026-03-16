#!/usr/bin/env bash
set -euo pipefail

OBJECT_CHECK_MODE="${1:-off}"
YOLO_CONF_THRES="${YOLO_CONF_THRES:-0.50}"
INSTRUCTION_INPUT_MODE="${INSTRUCTION_INPUT_MODE:-gui}"
INSTRUCTION_INPUT_FLAGS=()
OBJECT_CHECK_FLAGS=()
case "${INSTRUCTION_INPUT_MODE}" in
  stdin)
    ;;
  gui)
    INSTRUCTION_INPUT_FLAGS=(--instruction-gui)
    ;;
  *)
    echo "INSTRUCTION_INPUT_MODE must be stdin or gui (got: ${INSTRUCTION_INPUT_MODE})" >&2
    exit 1
    ;;
esac
# OBJECT_CHECK_MODE="${1:-always}"
case "${OBJECT_CHECK_MODE}" in
  off)
    OBJECT_CHECK_FLAGS=(--stdin-object-check-mode off)
    ;;
  not_found_only|always)
    OBJECT_CHECK_FLAGS=(--stdin-object-check --stdin-object-check-mode "${OBJECT_CHECK_MODE}")
    ;;
  *)
    echo "Usage: $0 [off|not_found_only|always]" >&2
    exit 1
    ;;
esac

# uv run scripts/demo_edge_server_visual.py \
echo "Runtime instruction update is enabled."
if [[ "${INSTRUCTION_INPUT_MODE}" == "gui" ]]; then
  echo "After start, update target noun phrase in the GUI window."
else
  echo "After start, type a new target noun phrase and press Enter in this terminal."
fi
echo "Object check mode: ${OBJECT_CHECK_MODE}"
echo "Instruction input mode: ${INSTRUCTION_INPUT_MODE}"
echo "Each instruction update updates target noun phrase for policy instruction."
echo "Mode=off: object check disabled (instruction updates only affect path generation)."
echo "Mode=not_found_only: periodic check only while NOT FOUND."
echo "Mode=always: periodic check always runs regardless of FOUND/NOT FOUND."
echo "If NOT FOUND, LeKiwi rotates in place until FOUND (only for non-off modes)."

client/.venv.client/bin/python scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --hef models/edge_adapter_accuracy_calib.hef \
    --instruction-verb "move to" \
    --instruction-noun "black bag" \
    "${INSTRUCTION_INPUT_FLAGS[@]}" \
    "${OBJECT_CHECK_FLAGS[@]}" \
    --yolo-hef models/yolo_world_v2s.hef \
    --clip-hef models/clip_vit_b_32_text_encoder.hef \
    --yolo-conf-thres "${YOLO_CONF_THRES}" \
    --task-mode language_only \
    --no-satellite \
    --output-format float32 \
    --show

# ssh -N -L 8000:0.0.0.0:8000 a100-highreso
