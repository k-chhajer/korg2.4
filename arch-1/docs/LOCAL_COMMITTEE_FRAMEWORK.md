# Local Committee Framework

## Short Answer

Yes. The current `arch-1` implementation is already close to a reusable framework for running local models as a fixed committee.

The reusable core is:

- config-driven role definitions
- OpenAI-compatible local backend support
- fixed stage execution
- role-specific system prompts
- role-specific output schemas
- role-specific local context views
- trace logging for every stage
- benchmark summaries and reports

The main missing step is packaging and hardening the interface so it feels like a standalone local framework instead of an eval harness.

## Existing Framework Shape

Current code path:

- `implementation/committee_llm/config.py`
  - loads provider, model, roles, schemas, context views, and architecture metadata
- `implementation/committee_llm/client.py`
  - calls any OpenAI-compatible `/v1/chat/completions` endpoint
- `implementation/committee_llm/orchestrator.py`
  - runs the fixed pipeline:
    - `coordinator_plan`
    - `researcher`
    - `analyst`
    - `critic`
    - `coordinator_finalize`
- `implementation/prompts/*.md`
  - role prompts
- `implementation/configs/*.json`
  - model/backend configs
- `implementation/committee_llm/benchmark.py`
  - batch runner
- `implementation/committee_llm/benchmark_report.py`
  - Markdown report generator

The local model requirement is simple: expose the model through an OpenAI-compatible endpoint. Ollama, vLLM, llama.cpp `llama-server`, LM Studio, or Modal/vLLM can all fit this pattern if they expose `/v1/chat/completions`.

## Minimal Local Run Pattern

Example with Ollama:

```bash
cd arch-1
export PYTHONPATH=implementation
export OPENAI_API_KEY=EMPTY
ollama serve
ollama pull qwen3:8b
python -m committee_llm.benchmark \
  --config implementation/configs/qwen3_8b_ollama_eval.json \
  --tasks evals/data/benchmarks/hotpotqa_dev_20.jsonl \
  --outdir evals/runs/hotpotqa_arch1_local \
  --system committee \
  --limit 20
```

Example config provider block:

```json
{
  "provider": {
    "api_base": "http://127.0.0.1:11434/v1",
    "api_key_env": "OPENAI_API_KEY",
    "timeout_sec": 600
  }
}
```

The same framework can target a local llama.cpp server:

```json
{
  "provider": {
    "api_base": "http://127.0.0.1:8080/v1",
    "api_key_env": "OPENAI_API_KEY",
    "timeout_sec": 600
  }
}
```

## What To Package

A clean standalone framework should expose three layers:

1. `CommitteeRunner`
   - Python API for one task.
   - Already exists in `orchestrator.py`.

2. `committee run`
   - CLI for one prompt or one JSONL task file.
   - Should write full traces and a final answer.

3. `committee bench`
   - Batch benchmark runner.
   - Already mostly exists as `committee_llm.benchmark`.

Suggested framework layout:

```text
committee_framework/
  config.py
  client.py
  runner.py
  traces.py
  evaluation.py
  cli.py
  prompts/
  configs/
```

Suggested Python API:

```python
from committee_llm.config import ExperimentConfig
from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.orchestrator import CommitteeRunner

config = ExperimentConfig.load("implementation/configs/qwen3_8b_ollama_eval.json")
client = OpenAICompatibleChatClient(
    api_base=config.provider.api_base,
    api_key_env=config.provider.api_key_env,
    timeout_sec=config.provider.timeout_sec,
)
runner = CommitteeRunner(config=config, client=client)

trace = runner.run("Explain why the sky is blue in one paragraph.")
print(trace["final_answer"])
```

## Paper-Valid vs Local Scaffold

The local framework can run today, but most current configs are prompt scaffolds, not paper-valid post-trained committees.

Prompt scaffold:

- one base model
- separate role prompts
- separate role schemas
- separate local context views
- same checkpoint reused across roles

Paper-valid Architecture 1:

- same fixed pipeline
- role-specific post-trained checkpoints
- no checkpoint reuse across roles
- `architecture.specialization_source = post_trained_role_specialists`

So the framework is valid as an execution system now. The stronger research claim needs actual role-specialized checkpoints.

## Measured Improvement Factors

### HotpotQA Distractor, First 75 Examples

Same first-75 HotpotQA examples, Qwen3-8B-AWQ on AWS g4dn.

| System | Completed | Failed | Answer EM | Answer F1 | Mean Sec/Task | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal Qwen single-shot | 75 | 0 | 0.0000 | 0.0162 | 23.06 | 174,866 |
| Committee | 73 | 2 | 0.3836 | 0.5774 | 150.04 | 1,543,424 |

Improvement:

| Metric | Calculation | Factor / Delta |
| --- | ---: | ---: |
| Answer F1 factor | `0.5774 / 0.0162` | `35.74x` |
| Answer F1 absolute gain | `0.5774 - 0.0162` | `+0.5613` |
| Answer EM absolute gain | `0.3836 - 0.0000` | `+0.3836` |
| Latency cost factor | `150.04 / 23.06` | `6.51x slower` |
| Token cost factor | `1,543,424 / 174,866` | `8.83x more tokens` |

Important caveat: the `35.74x` F1 factor is large because the single-shot baseline F1 was extremely close to zero. The absolute gain, `+0.5613`, is the more stable way to describe the improvement.

### GPQA Diamond

The 198-example GPQA committee run did not fully complete because of the Modal billing cap. It completed `163/198` examples.

| System / Reference | Accuracy |
| --- | ---: |
| Committee, completed traces only | 51.53% |
| Committee, conservative full-set lower bound | 42.42% |
| Qwen3-8B non-thinking official reference | 39.30% |
| Qwen3-8B thinking official reference | 62.00% |

Improvement against non-thinking Qwen3-8B official reference:

| Metric | Calculation | Factor / Delta |
| --- | ---: | ---: |
| Completed-trace accuracy factor | `51.53 / 39.30` | `1.31x` |
| Completed-trace absolute gain | `51.53 - 39.30` | `+12.23 percentage points` |
| Conservative lower-bound factor | `42.42 / 39.30` | `1.08x` |
| Conservative lower-bound absolute gain | `42.42 - 39.30` | `+3.12 percentage points` |

Comparison against thinking Qwen3-8B official reference:

| Metric | Calculation | Factor / Delta |
| --- | ---: | ---: |
| Completed-trace factor vs thinking | `51.53 / 62.00` | `0.83x` |
| Completed-trace gap vs thinking | `51.53 - 62.00` | `-10.47 percentage points` |
| Conservative lower-bound factor vs thinking | `42.42 / 62.00` | `0.68x` |
| Conservative lower-bound gap vs thinking | `42.42 - 62.00` | `-19.58 percentage points` |

Important caveat: GPQA is not a perfect apples-to-apples local single-shot comparison in the current artifacts. The clean claim is:

- committee beat the non-thinking Qwen3-8B official reference by `1.31x` on completed traces
- committee beat the same reference by `1.08x` even under the conservative full-set lower bound
- committee did not beat the official thinking Qwen3-8B reference

## Recommended Next Step

To make this a real local framework, do these in order:

1. Add a first-class `committee run` CLI for one-off local use.
2. Add a minimal task JSON schema for non-benchmark prompts.
3. Add a `configs/local_ollama.json` and `configs/local_llamacpp.json`.
4. Add a small README with Ollama, vLLM, and llama.cpp examples.
5. Add budget controls:
   - max total tokens per task
   - max wall time per stage
   - optional early stop if researcher or analyst fails schema validation
6. Add ablations:
   - no critic
   - no researcher
   - no analyst
   - budget-matched single-model baseline

That would turn Arch-1 from an eval scaffold into a reusable local committee runner.
