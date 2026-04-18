# Colab T4: Architecture 2 Controller Training

This runbook trains Architecture 2 only, inside `arch-2/`.

Training target:

- train only the controller policy/value network
- keep specialist LLM calls frozen and served through OpenRouter (`qwen/qwen3-8b`)

Persist outputs to Google Drive:

- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `train_log.jsonl`
- `eval_log.jsonl`
- `train_config.json`

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

## 3) Install Dependencies

```bash
%cd /content
!python3 -m pip install -U pip
!python3 -m pip install -U "torch>=2.2"
```

## 4) Set OpenRouter API Key

```python
import os
os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY"
```

## 5) (Optional) Quick API Connectivity Check

```python
import json
import time
import urllib.request
import os

url = "https://openrouter.ai/api/v1/models"
req = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
print("OpenRouter reachable. model_count=", len(payload.get("data", [])))
```

## 6) Output Directory

```python
OUTDIR = "/content/drive/MyDrive/korg2_4_runs/arch2_controller_hotpot"
print(OUTDIR)
```

## 7) Train Controller

```bash
%cd /content/grok_multiagent/arch-2/implementation
!python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json \
  --tasks /content/grok_multiagent/arch-2/evals/data/benchmarks/hotpotqa_dev.jsonl \
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

## 8) Resume

```bash
%cd /content/grok_multiagent/arch-2/implementation
!python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json \
  --tasks /content/grok_multiagent/arch-2/evals/data/benchmarks/hotpotqa_dev.jsonl \
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

## 9) Evaluate Best Checkpoint

```bash
%cd /content/grok_multiagent/arch-2/implementation
!python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json \
  --tasks /content/grok_multiagent/arch-2/evals/data/benchmarks/hotpotqa_dev.jsonl \
  --outdir "$OUTDIR" \
  --eval-task-count 24 \
  --device cuda \
  --resume "$OUTDIR/checkpoint_best.pt" \
  --evaluate-only
```

## 10) Check Progress

```bash
!ls -lah "$OUTDIR"
!tail -n 5 "$OUTDIR/train_log.jsonl"
!tail -n 5 "$OUTDIR/eval_log.jsonl"
```
