#!/bin/bash
set -e

cleanup() {
  echo "Finally: Clean up."
  rm -rf sam-vit-h-decoder-onnx-quantized
}

# When raise EXIT error, call cleanup using trap
trap 'cleanup' EXIT

echo "Download decoder model"
git lfs install
git clone https://huggingface.co/khsyee/sam-vit-h-decoder-onnx-quantized
mkdir -p checkpoint
cp sam-vit-h-decoder-onnx-quantized/sam_onnx_quantized.onnx checkpoint/
