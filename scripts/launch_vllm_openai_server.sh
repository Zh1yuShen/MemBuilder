#!/bin/bash
# Launch vLLM OpenAI-compatible server with auto TP by visible GPU count.
#
# Usage:
#   bash scripts/launch_vllm_openai_server.sh <model_path> [port] [tp_size|auto] [served_model_name]
#
# Examples:
#   bash scripts/launch_vllm_openai_server.sh /path/to/Qwen3-4B-Instruct-2507 8000 auto qwen3-4b
#   bash scripts/launch_vllm_openai_server.sh /path/to/model 9000 8 qwen3_235b_a22b
#
# Environment overrides:
#   VLLM_HOST=0.0.0.0
#   VLLM_DTYPE=bfloat16
#   VLLM_GPU_MEMORY_UTILIZATION=0.90
#   VLLM_MAX_MODEL_LEN=32768
#   VLLM_TRUST_REMOTE_CODE=1
#   VLLM_EXTRA_ARGS="--max-num-seqs 64 --enable-prefix-caching"

set -euo pipefail

usage() {
    echo "Usage: bash $0 <model_path> [port] [tp_size|auto] [served_model_name]"
}

MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
    usage
    exit 1
fi

PORT="${2:-8000}"
TP_INPUT="${3:-auto}"
SERVED_MODEL_NAME="${4:-$(basename "$MODEL_PATH")}"

HOST="${VLLM_HOST:-0.0.0.0}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
# Reduce noisy TensorFlow import logs from transformers.
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"

count_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" ]]; then
        local count
        count="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^\s*$/d' | wc -l | tr -d ' ')"
        if [[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )); then
            echo "$count"
            return
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )); then
            echo "$count"
            return
        fi
    fi

    echo "1"
}

GPU_COUNT="$(count_visible_gpus)"

if [[ "$TP_INPUT" == "auto" ]]; then
    TP_SIZE="$GPU_COUNT"
else
    TP_SIZE="$TP_INPUT"
fi

if ! [[ "$TP_SIZE" =~ ^[0-9]+$ ]] || (( TP_SIZE < 1 )); then
    echo "Error: tp_size must be a positive integer or 'auto', got: $TP_INPUT"
    exit 1
fi

echo "============================================================"
echo "Launching vLLM OpenAI Server"
echo "  model_path:         $MODEL_PATH"
echo "  served_model_name:  $SERVED_MODEL_NAME"
echo "  host:port:          $HOST:$PORT"
echo "  visible_gpus:       $GPU_COUNT"
echo "  tensor_parallel:    $TP_SIZE"
echo "  dtype:              $DTYPE"
echo "============================================================"

# Fast preflight: avoid opaque startup traceback on old transformers.
python - <<'PY'
import sys
try:
    import transformers
    import transformers.configuration_utils as cu
except Exception as e:
    print(f"Preflight failed: cannot import transformers ({e})")
    print("Try: python -m pip install -U 'transformers>=4.56,<5' 'tokenizers>=0.21.1'")
    raise SystemExit(2)

if not hasattr(cu, "layer_type_validation"):
    print(f"Preflight failed: transformers=={transformers.__version__} is too old for current vLLM.")
    print("Try: python -m pip install -U 'transformers>=4.56,<5' 'tokenizers>=0.21.1'")
    raise SystemExit(2)
PY

cmd=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
    cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${VLLM_EXTRA_ARGS} )
    cmd+=("${extra_args[@]}")
fi

exec "${cmd[@]}"
