#!/bin/bash
#
# Convert verl FSDP checkpoint to HuggingFace format.
#
# verl saves checkpoints in FSDP sharded format. This script merges them
# into a single HuggingFace-compatible model directory that can be loaded
# by vLLM or transformers.
#
# Requirements:
#   - Run on the SAME number of GPUs used during training (e.g., 16 GPUs → 16 GPUs)
#   - verl must be installed
#
# Usage:
#   bash scripts/convert_verl_to_hf.sh <checkpoint_dir> <output_dir>
#
# Example:
#   bash scripts/convert_verl_to_hf.sh \
#       checkpoints/membuilder/global_step_100 \
#       models/membuilder_step100_hf

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_dir  Path to verl checkpoint (e.g., checkpoints/xxx/global_step_100)"
    echo "  output_dir      Output directory for HuggingFace model"
    exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

echo "=========================================="
echo "Converting verl checkpoint to HuggingFace"
echo "=========================================="
echo "Input:  ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# verl provides a model_merger utility for FSDP → HF conversion
# See: https://verl.readthedocs.io/en/latest/advance/checkpoint.html
python -m verl.utils.model.model_merger \
    --backend fsdp \
    --hf_model_path "${CHECKPOINT_DIR}/actor" \
    --local_dir "${CHECKPOINT_DIR}" \
    --output_path "${OUTPUT_DIR}"

echo ""
echo "Conversion complete!"
echo "HuggingFace model saved to: ${OUTPUT_DIR}"
echo ""
echo "You can now use this model with vLLM:"
echo "  bash scripts/launch_vllm_openai_server.sh ${OUTPUT_DIR}"
