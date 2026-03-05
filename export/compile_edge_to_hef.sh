#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model.onnx> <output_dir> [hailo_arch] [calib_npy] [end_node_names]"
  echo "Example (strict, random calib): $0 edge_adapter.onnx ./build hailo10h"
  echo "Example (strict, explicit calib): $0 edge_adapter.onnx ./build hailo10h ./calib.npy"
  echo "Example (allow parser recommendation): ALLOW_PARSER_RECOMMENDATION=1 $0 edge_adapter.onnx ./build hailo10h"
  exit 1
fi

ONNX_PATH="$1"
OUT_DIR="$2"
HAILO_ARCH="${3:-hailo10h}"
CALIB_NPY="${4:-}"
END_NODES="${5:-action_chunk}"
ALLOW_PARSER_RECOMMENDATION="${ALLOW_PARSER_RECOMMENDATION:-0}"

mkdir -p "$OUT_DIR"
BASE_NAME="$(basename "${ONNX_PATH%.*}")"
HAR_PATH="$OUT_DIR/${BASE_NAME}.har"
QHAR_PATH="$OUT_DIR/${BASE_NAME}_quant.har"
HEF_PATH="$OUT_DIR/${BASE_NAME}.hef"

if ! command -v hailo >/dev/null 2>&1; then
  echo "Error: 'hailo' CLI not found. Install Hailo Dataflow Compiler first."
  exit 2
fi

echo "[1/3] Parsing ONNX -> HAR"
PARSER_ARGS=(parser onnx "$ONNX_PATH" --hw-arch "$HAILO_ARCH" --har-path "$HAR_PATH" --end-node-names "$END_NODES")
if [[ "$ALLOW_PARSER_RECOMMENDATION" == "1" ]]; then
  PARSER_ARGS+=(-y)
fi
hailo "${PARSER_ARGS[@]}"

echo "[2/3] Quantizing HAR -> Quantized HAR"
if [[ -n "$CALIB_NPY" ]]; then
  hailo optimize "$HAR_PATH" --hw-arch "$HAILO_ARCH" --calib-set-path "$CALIB_NPY" --output-har-path "$QHAR_PATH"
else
  hailo optimize "$HAR_PATH" --hw-arch "$HAILO_ARCH" --use-random-calib-set --output-har-path "$QHAR_PATH"
fi

echo "[3/3] Compiling Quantized HAR -> HEF"
hailo compiler "$QHAR_PATH" --hw-arch "$HAILO_ARCH" --output-dir "$OUT_DIR"

if [[ ! -f "$HEF_PATH" ]]; then
  echo "Error: compile finished but HEF not found: $HEF_PATH"
  exit 3
fi

echo "Compiled HEF: $HEF_PATH"
