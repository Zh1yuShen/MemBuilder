#!/bin/bash
# ============================================
# MemBuilder SFT Training Example
# ============================================
# This script trains a memory construction model using LLaMA-Factory.
# Requires: LLaMA-Factory installed and SFT data prepared.
#
# Usage:
#   1. Set LLAMA_FACTORY_DIR to your LLaMA-Factory installation
#   2. Prepare SFT data using scripts/convert_trajectories_to_sft.py
#   3. Run this script

set -e

# ============================================
# Configuration (MODIFY THESE)
# ============================================

# Path to LLaMA-Factory installation
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/path/to/LLaMA-Factory}"

# Base model (Qwen3-4B recommended)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

# Dataset name (must be registered in dataset_info.json)
DATASET_NAME="${DATASET_NAME:-memory_building_sft}"

# Output directory for trained model
OUTPUT_DIR="${OUTPUT_DIR:-${LLAMA_FACTORY_DIR}/saves/membuilder-sft}"

# Training hyperparameters
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
CUTOFF_LEN="${CUTOFF_LEN:-20000}"

# ============================================
# Validation
# ============================================

if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "ERROR: LLaMA-Factory not found at $LLAMA_FACTORY_DIR"
    echo "Please set LLAMA_FACTORY_DIR environment variable"
    exit 1
fi

if [ ! -f "$LLAMA_FACTORY_DIR/data/dataset_info.json" ]; then
    echo "ERROR: dataset_info.json not found"
    exit 1
fi

# Check if dataset is registered
if ! grep -q "\"$DATASET_NAME\"" "$LLAMA_FACTORY_DIR/data/dataset_info.json"; then
    echo "ERROR: Dataset '$DATASET_NAME' not found in dataset_info.json"
    echo "Please register your dataset first"
    exit 1
fi

# ============================================
# Run Training
# ============================================

echo "============================================"
echo "MemBuilder SFT Training"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "============================================"

cd "$LLAMA_FACTORY_DIR"

llamafactory-cli train \
    --model_name_or_path "$MODEL_PATH" \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ds_z2_config.json \
    --dataset "$DATASET_NAME" \
    --template qwen3 \
    --cutoff_len "$CUTOFF_LEN" \
    --overwrite_cache \
    --resize_vocab \
    --preprocessing_num_workers 16 \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.1 \
    --bf16 \
    --ddp_timeout 180000000

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To export model:"
echo "  cp $OUTPUT_DIR/*.safetensors /path/to/deployed_model/"
echo "  cp $OUTPUT_DIR/*.json /path/to/deployed_model/"
