# Committee LLM Harness

This repo scaffolds Architecture 1 for your paper: explicit multi-agent coordination using separate role-specialized instances of the same base model or different models per role.

The default pipeline is:

1. `coordinator_plan`
2. `researcher`
3. `analyst`
4. `critic`
5. `coordinator_finalize`

The coordinator is reused twice, which matches your idea of a fixed flow with explicit orchestration rather than latent internal deliberation.

## What This Gives You

- A minimal Python CLI for running the committee on a single prompt or a JSONL task file
- Swappable OpenAI-compatible backends, so you can point it at local vLLM, LM Studio, OpenRouter, or another compatible server
- Separate prompt files for each role so you can tune the behavioral signature of the architecture without changing code
- Full JSON traces for every stage, which is important for an architecture paper because you want to analyze the intermediate behavior, not just the final answer
- Aggregated run-level usage summaries plus batch `summary.json` manifests for experiment analysis
- Structured research-task specs with context, deliverables, evaluation criteria, and deterministic answer checks

## Project Layout

- Implementation:
[committee_llm/cli.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/cli.py)
- Implementation:
[committee_llm/orchestrator.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/orchestrator.py)
- Implementation:
[committee_llm/client.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/client.py)
- Implementation:
[committee_llm/config.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/config.py)
- Implementation:
[committee_llm/tasks.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/tasks.py)
- Implementation:
[committee_llm/evaluation.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/evaluation.py)
- Evaluation tooling:
[committee_llm/benchmark_data.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/benchmark_data.py)
- Evaluation tooling:
[committee_llm/benchmark.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/benchmark.py)
- Evaluation tooling:
[committee_llm/baselines.py](/C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/baselines.py)
- Configs:
[qwen3_8b_committee.json](/C:/Users/luthi/Documents/korg/arch-1/implementation/configs/qwen3_8b_committee.json)
- Prompts:
[coordinator_plan.md](/C:/Users/luthi/Documents/korg/arch-1/implementation/prompts/coordinator_plan.md)
- Prompts:
[researcher.md](/C:/Users/luthi/Documents/korg/arch-1/implementation/prompts/researcher.md)
- Prompts:
[analyst.md](/C:/Users/luthi/Documents/korg/arch-1/implementation/prompts/analyst.md)
- Prompts:
[critic.md](/C:/Users/luthi/Documents/korg/arch-1/implementation/prompts/critic.md)
- Prompts:
[coordinator_finalize.md](/C:/Users/luthi/Documents/korg/arch-1/implementation/prompts/coordinator_finalize.md)
- Evals data:
[data](/C:/Users/luthi/Documents/korg/arch-1/evals/data)
- Evals runs:
[runs](/C:/Users/luthi/Documents/korg/arch-1/evals/runs)
- Evals tests:
[tests](/C:/Users/luthi/Documents/korg/arch-1/evals/tests)

## Quick Start

Set the backend environment variables:

```powershell
cd C:\Users\luthi\Documents\korg\arch-1
$env:PYTHONPATH = "implementation"
$env:OPENAI_API_KEY = "EMPTY"
```

The API base URL comes from the config JSON, not `OPENAI_API_BASE`.

Run one prompt:

```powershell
python -m committee_llm --config implementation\configs\qwen3_8b_committee.json --prompt "Explain whether explicit coordination can outperform a single strong model on multi-hop reasoning."
```

Run one structured research task:

```powershell
python -m committee_llm --config implementation\configs\qwen3_8b_committee.json --task evals\data\sample_research_task.json --output evals\runs\research_task.json
```

Save a single trace:

```powershell
python -m committee_llm --config implementation\configs\qwen3_8b_committee.json --prompt "Draft a research hypothesis for black-box architecture inference." --output evals\runs\single_trace.json
```

Run a batch from JSONL:

```powershell
python -m committee_llm --config implementation\configs\qwen3_8b_committee.json --tasks evals\data\sample_tasks.jsonl --outdir evals\runs\sample_batch
```

Keep a long batch running even if one task fails:

```powershell
python -m committee_llm --config implementation\configs\qwen3_8b_committee.json --tasks evals\data\sample_tasks.jsonl --outdir evals\runs\sample_batch --continue-on-error
```

Each JSONL line should look like:

```json
{"id": "task-001", "prompt": "Your benchmark question here"}
```

For research tasks, you can provide richer structure:

```json
{
  "id": "research-001",
  "domain": "llm_evaluation",
  "prompt": "Design an experiment that isolates coordination gains from extra inference compute.",
  "context": "You are writing for an architecture paper. Assume access to the same base model under different orchestration schemes.",
  "objectives": [
    "Define treatment and control conditions",
    "Keep total inference budget comparable",
    "Name primary metrics and likely confounders"
  ],
  "constraints": [
    "Do not assume proprietary tooling",
    "Keep the proposal feasible for a small academic lab"
  ],
  "deliverable": "A concise experimental design note with explicit controls and metrics.",
  "evaluation_criteria": [
    "Explains how compute is matched across conditions",
    "Includes at least one baseline",
    "Identifies concrete evaluation metrics"
  ],
  "checks": {
    "must_contain": ["baseline", "compute", "metric"],
    "min_words": 120
  },
  "metadata": {
    "split": "dev"
  }
}
```

Batch runs write one trace per task plus `summary.json`, which aggregates wall-clock time, run time, token usage, deterministic evaluation scores, and domain-level breakdowns.

For publication-style comparisons between the committee and a single-pass baseline on canonical benchmarks, see [BENCHMARKS.md](/C:/Users/luthi/Documents/korg/arch-1/docs/BENCHMARKS.md).

## Configuring Same-Model vs Multi-Model Committees

The default config uses one model name for every role. To run separate models, set the `model` field inside individual roles in [qwen3_8b_committee.json](/C:/Users/luthi/Documents/korg/arch-1/implementation/configs/qwen3_8b_committee.json).

Example:

```json
"researcher": {
  "system_prompt_path": "../prompts/researcher.md",
  "model": "Qwen/Qwen3-30B-A3B-Instruct",
  "temperature": 0.4,
  "max_tokens": 1400
}
```

That lets you compare:

- Same model + different role prompts
- Shared weights + conditional role prompting
- Heterogeneous committees

## Why This Matches Your Paper

This scaffold cleanly isolates explicit coordination as the treatment variable:

- Specialization is induced externally through prompts and fixed routing
- Coordination is observable in the trace
- The final answer is produced after explicit critique and merge
- You can later compare this against a single-model internalized deliberation baseline using the same tasks and output format

## Benchmarking Suggestions

Since this is an architecture paper, do not only benchmark final accuracy. Track:

- Final task score
- Total latency
- Total token usage
- Per-stage token usage
- Agreement or disagreement between stages
- Critic revision pressure: how often the critic forces substantial changes
- Trace signatures: verbosity, decomposition depth, number of cited assumptions, number of detected errors

Good benchmark families for this setup:

- Multi-hop QA: HotpotQA, MuSiQue
- Reasoning: GSM8K, MATH, BBH
- Knowledge-intensive evaluation: GPQA
- Open-ended synthesis: short research-planning prompts scored by an LLM judge and by manual rubric

For the paper framing, the cleanest comparison is:

1. Architecture 1: explicit committee with routing
2. Architecture 2: shared-weight conditional roles with the same fixed pipeline
3. Architecture 3: single-call or iterative single-model deliberation with internalized roles

## Notes

- The code uses only the Python standard library.
- The backend must expose an OpenAI-compatible `chat/completions` endpoint.
- Local servers usually accept `OPENAI_API_KEY=EMPTY`.
