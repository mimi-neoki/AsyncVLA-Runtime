#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model.onnx> <calib_dir> <output_dir> [hailo_arch]"
  echo "Example: $0 edge_adapter.onnx ./calib ./build hailo10h"
  exit 1
fi

ONNX_PATH="$1"
CALIB_DIR="$2"
OUT_DIR="$3"
HAILO_ARCH="${4:-hailo10h}"

mkdir -p "$OUT_DIR"
BASE_NAME="$(basename "${ONNX_PATH%.*}")"
HAR_PATH="$OUT_DIR/${BASE_NAME}.har"
QHAR_PATH="$OUT_DIR/${BASE_NAME}_quant.har"
HEF_PATH="$OUT_DIR/${BASE_NAME}.hef"

if ! command -v hailo >/dev/null 2>&1; then
  echo "Error: 'hailo' CLI not found. Install Hailo Dataflow Compiler first."
  exit 1
fi

echo "[1/3] Parsing ONNX -> HAR"
hailo parser onnx "$ONNX_PATH" --hw-arch "$HAILO_ARCH" --output-har "$HAR_PATH"

echo "[2/3] Quantizing HAR -> Quantized HAR"
hailo optimize "$HAR_PATH" --hw-arch "$HAILO_ARCH" --calib-set-path "$CALIB_DIR" --output-har "$QHAR_PATH"

echo "[3/3] Compiling Quantized HAR -> HEF"
hailo compiler "$QHAR_PATH" --hw-arch "$HAILO_ARCH" --hef "$HEF_PATH"

echo "Compiled HEF: $HEF_PATH"
