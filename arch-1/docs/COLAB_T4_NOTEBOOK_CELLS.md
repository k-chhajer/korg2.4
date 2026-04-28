# Colab Notebook Cells for GPQA Committee on T4

These cells run the `committee` GPQA Diamond benchmark on a Colab T4 using a local OpenAI-compatible vLLM server and write outputs directly to Google Drive so they survive timeouts.

Important:

- this is the same benchmark family and the same repo harness as the Modal run
- it is not bit-identical to the Modal setup, because a T4 is much more practical with `Qwen/Qwen3-8B-AWQ` than the full unquantized checkpoint
- the benchmark runner now checkpoints both per-task traces and `summary.json` after every iteration, so rerunning with `--resume` continues where it left off

## Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 2: Clone Repo

```bash
%cd /content
!rm -rf /content/korg2.4
!git clone YOUR_REPO_URL /content/korg2.4
```

## Cell 3: Install vLLM

```bash
%cd /content
!python3 -m pip install -U pip
!python3 -m pip install -U "vllm>=0.8.5"
```

## Cell 4: Start the Local OpenAI-Compatible Server

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

## Cell 5: Wait for the Server

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

## Cell 6: Set Drive Output Directory

```python
OUTDIR = "/content/drive/MyDrive/korg2_4_runs/gpqa_diamond_colab_t4_committee"
print(OUTDIR)
```

## Cell 7: Run the GPQA Diamond Committee Benchmark

```bash
%cd /content/korg2.4/arch-1/implementation
!OPENAI_API_KEY=localtest python3.11 -m committee_llm.benchmark \
  --config /content/korg2.4/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json \
  --tasks /content/korg2.4/arch-1/evals/data/benchmarks/gpqa_diamond.jsonl \
  --outdir "$OUTDIR" \
  --system committee \
  --continue-on-error \
  --resume
```

## Cell 8: Inspect Progress or Resume Later

```bash
!ls -lah "$OUTDIR"
!find "$OUTDIR/committee" -maxdepth 1 -name '*.json' | wc -l
!python3 -m json.tool "$OUTDIR/summary.json" | head -n 80
```

If Colab disconnects, remount Drive, reclone the repo, restart the local server, and rerun Cell 7 exactly as-is. Because the runner writes each example trace and checkpoints `summary.json` after every task, `--resume` will skip completed examples.
