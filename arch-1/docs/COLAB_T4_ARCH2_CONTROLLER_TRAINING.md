# Colab T4: Architecture 2 Controller Training

This runbook trains the Architecture 2 controller policy (RL over frozen specialists) on Colab.

What is trained:

- only the controller policy/value network
- specialist role calls remain frozen and are served by a local OpenAI-compatible vLLM endpoint

What is persisted:

- checkpoints (`checkpoint_last.pt`, `checkpoint_best.pt`)
- training logs (`train_log.jsonl`, `eval_log.jsonl`)
- config snapshot (`train_config.json`)

Write output to Google Drive so runs survive Colab disconnects.

## 1) Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2) Clone Repo

```bash
%cd /content
!rm -rf /content/grok_multiagent
!git clone YOUR_REPO_URL /content/grok_multiagent
```

## 3) Install Runtime Dependencies

```bash
%cd /content
!python3 -m pip install -U pip
!python3 -m pip install -U "vllm>=0.8.5" "torch>=2.2"
```

## 4) Start Local OpenAI-Compatible Model Server

```bash
%cd /content
!nohup python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key localtest \
  --model Qwen/Qwen3-8B-AWQ \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  > /content/vllm_qwen3_8b_awq.log 2>&1 &
```

## 5) Wait for Server Ready

```python
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/models"
deadline = time.time() + 900

while True:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        print("Server is ready")
        print(payload)
        break
    except Exception as exc:
        if time.time() > deadline:
            raise RuntimeError("Timed out waiting for vLLM server") from exc
        print("Waiting for server...", exc)
        time.sleep(10)
```

## 6) Choose Output Directory (Drive)

```python
OUTDIR = "/content/drive/MyDrive/korg2_4_runs/arch2_controller_hotpot"
print(OUTDIR)
```

## 7) Train Controller (Architecture 2)

This command uses the existing frozen committee config and trains only the controller.

```bash
%cd /content/grok_multiagent/arch-1/implementation
!OPENAI_API_KEY=localtest python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json \
  --tasks /content/grok_multiagent/arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl \
  --outdir "$OUTDIR" \
  --episodes 120 \
  --eval-every 10 \
  --save-every 5 \
  --eval-task-count 24 \
  --budget-tokens 16000 \
  --max-decisions 6 \
  --max-restarts 1 \
  --per-role-call-cap 2 \
  --device cuda
```

## 8) Resume After Colab Timeout

Use the same command with `--resume`:

```bash
%cd /content/grok_multiagent/arch-1/implementation
!OPENAI_API_KEY=localtest python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json \
  --tasks /content/grok_multiagent/arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl \
  --outdir "$OUTDIR" \
  --episodes 120 \
  --eval-every 10 \
  --save-every 5 \
  --eval-task-count 24 \
  --budget-tokens 16000 \
  --max-decisions 6 \
  --max-restarts 1 \
  --per-role-call-cap 2 \
  --device cuda \
  --resume "$OUTDIR/checkpoint_last.pt"
```

## 9) Evaluate Best Checkpoint Only

```bash
%cd /content/grok_multiagent/arch-1/implementation
!OPENAI_API_KEY=localtest python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json \
  --tasks /content/grok_multiagent/arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl \
  --outdir "$OUTDIR" \
  --eval-task-count 24 \
  --device cuda \
  --resume "$OUTDIR/checkpoint_best.pt" \
  --evaluate-only
```

## 10) Track Progress

```bash
!ls -lah "$OUTDIR"
!tail -n 5 "$OUTDIR/train_log.jsonl"
!tail -n 5 "$OUTDIR/eval_log.jsonl"
```

## Optional: Run MuSiQue / GPQA

The controller trainer accepts any task JSONL supported by this repo. You can pass:

- `evals/data/benchmarks/musique_*.jsonl` (if present)
- `evals/data/benchmarks/gpqa_diamond.jsonl`

Use the same command shape with a different `--tasks` path.
