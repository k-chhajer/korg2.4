#!/bin/bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

pkill -f /home/ubuntu/run_gpqa_aws_use1.sh || true
pkill -f vllm.entrypoints.openai.api_server || true

timeout 180 python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen3-8B-AWQ \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16
