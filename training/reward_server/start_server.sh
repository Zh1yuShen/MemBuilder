#!/bin/bash
# Open-source version: Start reward server with standard OpenAI API
# Usage: OPENAI_API_KEY=xxx ./start_server.sh [port]

PORT=${1:-8765}
HOST=${2:-0.0.0.0}

# Check required environment variables
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is required"
    echo "Usage: OPENAI_API_KEY=xxx ./start_server.sh [port]"
    exit 1
fi

# Optional: custom base URL
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

echo "============================================"
echo "Starting Reward Server (OpenAI)"
echo "============================================"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "API Base: ${OPENAI_BASE_URL}"
echo "============================================"

cd "$(dirname "$0")/../.."

OPENAI_API_KEY="${OPENAI_API_KEY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
python -m training.reward_server.server --host ${HOST} --port ${PORT}
