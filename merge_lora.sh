#!/bin/bash

# Merging LoRA weights into the base model
# Source Checkpoint: v12 (Step 3189)
# Target Model Directory: Qwen3_Chart_Stage1

echo "Start merging LoRA weights..."

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CKPT_DIR="../svg_output/v12-20260102-163107/checkpoint-3189"
OUTPUT_DIR="models/Qwen3_Chart_Stage1"

cd "$SCRIPT_DIR" || exit 1

echo "Source: $CKPT_DIR"
echo "Target: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "$CKPT_DIR" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

echo "Merge completed!"
