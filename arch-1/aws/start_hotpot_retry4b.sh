#!/bin/bash
set -euo pipefail

pkill -f run_gpqa_aws_use1.sh || true
pkill -f committee_llm.benchmark || true
pkill -f vllm.entrypoints.openai.api_server || true

if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the environment." >&2
  exit 1
fi
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION=us-east-1
export ROOT_DIR=/home/ubuntu/arch-1
export ARCHIVE_PATH=/home/ubuntu/arch1-aws-benchmark-20260407c.tar.gz
export RUN_ID=hotpotqa_aws_g4dn_thinking_75_20260407_retry4b
export TASKS_PATH=/home/ubuntu/arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl
export LIMIT_ARG=75
export BUCKET_URI=s3://korg24-arch1-gpqa-20260406-525184038455

chmod +x /home/ubuntu/run_gpqa_aws_use1.sh
nohup /home/ubuntu/run_gpqa_aws_use1.sh >/home/ubuntu/run_hotpotqa_75_retry4b_launcher.log 2>&1 </dev/null &
