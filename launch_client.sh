#!/usr/bin/env bash
set -euo pipefail

OBJECT_CHECK_MODE="${1:-off}"
YOLO_CONF_THRES="${YOLO_CONF_THRES:-0.50}"
INSTRUCTION_INPUT_MODE="${INSTRUCTION_INPUT_MODE:-gui}"
EDGE_BACKEND="${EDGE_BACKEND:-hef}"
EDGE_DEVICE="${EDGE_DEVICE:-cpu}"
EDGE_DTYPE="${EDGE_DTYPE:-float32}"
EDGE_HEF_PATH="${EDGE_HEF_PATH:-build_fixed/edge_adapter_balanced_hfimg_fixedp99_0_full1024.hef}"
EDGE_HF_DIR="${EDGE_HF_DIR:-$HOME/gitrepo/AsyncVLA_release}"
EDGE_FUSED_OUTPUT_NAME="${EDGE_FUSED_OUTPUT_NAME:-fused_feature}"
EDGE_FUSED_DIM="${EDGE_FUSED_DIM:-1024}"
TOKEN_QUANT_MODE="${TOKEN_QUANT_MODE:-fixed_affine}"
TOKEN_QUANT_PARAMS="${TOKEN_QUANT_PARAMS:-calib_data/token_quant_fixed_affine_p99_0_uint8.npz}"
INSTRUCTION_INPUT_FLAGS=()
OBJECT_CHECK_FLAGS=()
EDGE_BACKEND_FLAGS=()
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

case "${EDGE_BACKEND}" in
  hef)
    EDGE_BACKEND_FLAGS=(
      --edge-backend hef
      --hef "${EDGE_HEF_PATH}"
      --token-quant-mode "${TOKEN_QUANT_MODE}"
      --token-quant-params "${TOKEN_QUANT_PARAMS}"
    )
    ;;
  hf)
    EDGE_BACKEND_FLAGS=(
      --edge-backend hf
      --hf-dir "${EDGE_HF_DIR}"
      --edge-device "${EDGE_DEVICE}"
      --edge-dtype "${EDGE_DTYPE}"
      --token-quant-mode "${TOKEN_QUANT_MODE}"
      --token-quant-params "${TOKEN_QUANT_PARAMS}"
    )
    ;;
  hef_torch_head)
    EDGE_BACKEND_FLAGS=(
      --edge-backend hef_torch_head
      --hef "${EDGE_HEF_PATH}"
      --hf-dir "${EDGE_HF_DIR}"
      --edge-device "${EDGE_DEVICE}"
      --edge-dtype "${EDGE_DTYPE}"
      --output-fused-name "${EDGE_FUSED_OUTPUT_NAME}"
      --fused-dim "${EDGE_FUSED_DIM}"
      --token-quant-mode "${TOKEN_QUANT_MODE}"
      --token-quant-params "${TOKEN_QUANT_PARAMS}"
    )
    ;;
  *)
    echo "EDGE_BACKEND must be hef, hf, or hef_torch_head (got: ${EDGE_BACKEND})" >&2
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
echo "Edge backend: ${EDGE_BACKEND}"
if [[ "${EDGE_BACKEND}" == "hef" ]]; then
  echo "Edge HEF: ${EDGE_HEF_PATH}"
elif [[ "${EDGE_BACKEND}" == "hef_torch_head" ]]; then
  echo "Edge HEF: ${EDGE_HEF_PATH}"
  echo "Edge HF dir: ${EDGE_HF_DIR}"
  echo "Edge device: ${EDGE_DEVICE}"
  echo "Edge dtype: ${EDGE_DTYPE}"
  echo "Fused output: ${EDGE_FUSED_OUTPUT_NAME}"
  echo "Fused dim: ${EDGE_FUSED_DIM}"
else
  echo "Edge HF dir: ${EDGE_HF_DIR}"
  echo "Edge device: ${EDGE_DEVICE}"
  echo "Edge dtype: ${EDGE_DTYPE}"
fi
echo "Token quant mode: ${TOKEN_QUANT_MODE}"
echo "Token quant params: ${TOKEN_QUANT_PARAMS}"
echo "Each instruction update updates target noun phrase for policy instruction."
echo "Mode=off: object check disabled (instruction updates only affect path generation)."
echo "Mode=not_found_only: periodic check only while NOT FOUND."
echo "Mode=always: periodic check always runs regardless of FOUND/NOT FOUND."
echo "If NOT FOUND, LeKiwi rotates in place until FOUND (only for non-off modes)."

client/.venv.client/bin/python scripts/demo_lekiwi_client.py \
    --policy-url http://0.0.0.0:8000/infer \
    --instruction-verb "move to" \
    --instruction-noun "black bag" \
    "${EDGE_BACKEND_FLAGS[@]}" \
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
