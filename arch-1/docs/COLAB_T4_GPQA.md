# Colab T4 GPQA Runbook

This runbook is for running the same `committee` GPQA Diamond benchmark on a Colab T4 runtime without Modal.

## What already persists

The benchmark runner now writes:

- each per-example trace immediately to `OUTDIR/committee/<task_id>.json`
- a rolling `OUTDIR/summary.json` checkpoint after every completed task
- a rolling `OUTDIR/summary.json` checkpoint after every failed task

So if Colab times out, rerun with the same `--outdir` and `--resume` and it will reuse existing task outputs.

## Recommended output root

Write directly to Drive so outputs survive runtime resets:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Suggested output root:

```bash
/content/drive/MyDrive/korg2_4_runs/gpqa_diamond_colab_t4_committee
```

## Repo setup

```bash
cd /content
git clone <your-repo-url> korg2.4
cd /content/korg2.4/arch-1
python3.11 -m pip install -U pip
```

This repo does not require third-party Python packages for the benchmark client itself.

## Start the local model server

You need an OpenAI-compatible server on `127.0.0.1:8000`.

The checked-in Colab config expects:

- config: `/content/korg2.4/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json`
- API base: `http://127.0.0.1:8000/v1`

One workable pattern is a Colab-hosted Qwen3-8B server that exposes the OpenAI Chat Completions API. The exact server stack is your choice, but it must support the request body used by the repo config, including:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

for the two coordinator stages.

## Run the GPQA committee benchmark

```bash
cd /content/korg2.4/arch-1/implementation
OPENAI_API_KEY=localtest \
python3.11 -m committee_llm.benchmark \
  --config /content/korg2.4/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json \
  --tasks /content/korg2.4/arch-1/evals/data/benchmarks/gpqa_diamond.jsonl \
  --outdir /content/drive/MyDrive/korg2_4_runs/gpqa_diamond_colab_t4_committee \
  --system committee \
  --continue-on-error \
  --resume
```

## Resume after timeout

Use the exact same command again. Because each task trace and the rolling summary are written to Drive, the runner will skip completed examples and continue from the remaining ones.

## Notes

- A full `198`-example committee run on a T4 is possible but likely slow.
- If memory is tight, use a quantized backend or lower-concurrency server settings.
- If the local server does not understand `chat_template_kwargs.enable_thinking`, disable that in the config or adapt the server layer to translate it.
