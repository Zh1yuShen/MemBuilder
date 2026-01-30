#!/bin/bash
# Launch vLLM OpenAI-compatible server
# Usage: bash launch_vllm_openai_server.sh <model_path> [port] [tp_size]

python -m vllm.entrypoints.openai.api_server \
    --model "${1:?Usage: $0 <model_path> [port] [tp_size]}" \
    --host 0.0.0.0 \
    --port "${2:-8000}" \
    --tensor-parallel-size "${3:-1}" \
    --trust-remote-code \
    --dtype bfloat16
