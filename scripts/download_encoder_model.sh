#!/bin/bash
set -e

cleanup() {
  echo "Finally: Clean up."
  rm -rf sam-vit-h-encoder-torchscript
}

# When raise EXIT error, call cleanup using trap
trap 'cleanup' EXIT

echo "Download encoder model"
git lfs install
git clone https://huggingface.co/khsyee/sam-vit-h-encoder-torchscript
cp -r sam-vit-h-encoder-torchscript/model_repository model_repository
