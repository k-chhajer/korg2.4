#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ubuntu/arch-1}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/home/ubuntu/arch1-aws-benchmark-20260406.tar.gz}"
RUN_ID="${RUN_ID:-gpqa_aws_g4dn_thinking_198_20260406}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/evals/runs/$RUN_ID}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/implementation/configs/qwen3_8b_colab_t4_eval.json}"
TASKS_PATH="${TASKS_PATH:-$ROOT_DIR/evals/data/benchmarks/gpqa_diamond.jsonl}"
SYSTEM_NAME="${SYSTEM_NAME:-committee}"
LIMIT_ARG="${LIMIT_ARG:-}"
BUCKET_URI="${BUCKET_URI:-}"
BENCH_LOG="$RUN_DIR/benchmark.log"
VLLM_LOG="$RUN_DIR/vllm.log"

cd /home/ubuntu
rm -rf "$ROOT_DIR"
tar -xzf "$ARCHIVE_PATH"

cd "$ROOT_DIR"
mkdir -p "$RUN_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install \
  "huggingface_hub==0.36.2" \
  "hf_transfer>=0.1.9" \
  "transformers==4.56.1" \
  "vllm==0.10.2"

export HF_HUB_ENABLE_HF_TRANSFER=1
export OPENAI_API_KEY=localtest
export PYTHONPATH="$ROOT_DIR/implementation${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
export VLLM_USE_V1=0

# Make the local package importable even if the shell/runtime drops PYTHONPATH.
PY_SITE_DIR="$(python3 - <<'PY'
import site
paths = site.getusersitepackages()
print(paths if isinstance(paths, str) else paths[0])
PY
)"
mkdir -p "$PY_SITE_DIR"
ln -sfn "$ROOT_DIR/implementation/committee_llm" "$PY_SITE_DIR/committee_llm"

nohup python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen3-8B-AWQ \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --disable-frontend-multiprocessing \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  >"$VLLM_LOG" 2>&1 &

for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 10
done

if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "vLLM server failed to become ready" | tee -a "$BENCH_LOG"
  exit 1
fi

if [ -n "$BUCKET_URI" ]; then
  (
    while true; do
      aws s3 sync "$RUN_DIR" "$BUCKET_URI/$RUN_ID/" --exclude "*.tmp" || true
      sleep 300
    done
  ) &
  SYNC_PID=$!
else
  SYNC_PID=""
fi

set +e
cd "$ROOT_DIR/implementation"
{
  echo "PWD=$(pwd)"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "implementation_dir_listing:"
  ls -la "$ROOT_DIR/implementation" || true
  echo "committee_llm_dir_listing:"
  ls -la "$ROOT_DIR/implementation/committee_llm" || true
  python3 - <<'PY'
import os
import sys
print("python_executable=", sys.executable)
print("sys_path=", sys.path)
import committee_llm
import committee_llm.benchmark
print("committee_llm_import_ok=", committee_llm.__file__)
PY
} >>"$BENCH_LOG" 2>&1
PRECHECK_EXIT=$?
if [ "$PRECHECK_EXIT" -ne 0 ]; then
  BENCH_EXIT=$PRECHECK_EXIT
  set -e
else
BENCH_CMD=(
  python3 -m committee_llm.benchmark
  --config "$CONFIG_PATH"
  --tasks "$TASKS_PATH"
  --outdir "$RUN_DIR"
  --system "$SYSTEM_NAME"
  --continue-on-error
)
if [ -n "$LIMIT_ARG" ]; then
  BENCH_CMD+=(--limit "$LIMIT_ARG")
fi
"${BENCH_CMD[@]}" >"$BENCH_LOG" 2>&1
BENCH_EXIT=$?
set -e
fi

if [ -n "${SYNC_PID:-}" ]; then
  kill "$SYNC_PID" || true
fi

if [ -n "$BUCKET_URI" ]; then
  aws s3 sync "$RUN_DIR" "$BUCKET_URI/$RUN_ID/" --exclude "*.tmp" || true
  tar -czf "/home/ubuntu/${RUN_ID}.tar.gz" -C "$ROOT_DIR/evals/runs" "$RUN_ID"
  aws s3 cp "/home/ubuntu/${RUN_ID}.tar.gz" "$BUCKET_URI/${RUN_ID}.tar.gz" || true
fi

sudo shutdown -h now || true
exit "$BENCH_EXIT"
