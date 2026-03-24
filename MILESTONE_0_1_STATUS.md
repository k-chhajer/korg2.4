# Milestone 0/1 Starter Status

Implemented in this pass:

1. Milestone 0
- Frozen protocol spec: `configs/protocol/protocol.yaml`
- Experiment registry with run IDs: `configs/experiments/registry.yaml`
- Deterministic eval seed policy and starter eval dataset:
  - `configs/eval/eval_config.yaml`
  - `data/processed/eval_suite.jsonl`

2. Milestone 1
- Native runtime orchestration graph:
  - `src/orchestrator/pipeline.py`
  - `src/orchestrator/backend.py`
  - `src/orchestrator/types.py`
- Sequential first-pass runtime:
  - captain decomposes and assigns tasks
  - Harper plans web queries, executes search, and writes grounded findings
  - Benjamin performs rigorous reasoning
  - Lucas reviews Harper and Benjamin, then emits a structured critique
  - captain synthesizes the final answer
- Structured critique validation and retry handling:
  - `src/critique/validator.py`
- Prompt protocol and role templates:
  - `prompts/protocol/*`
  - `prompts/roles/*`
- Runtime backend and search configuration:
  - `configs/model/runtime.yaml`
- Search and parsing utilities:
  - `src/tools/web_search.py`
  - `src/tools/json_utils.py`
- Logging and token proxy utilities:
  - `src/tools/run_logger.py`
  - `src/tools/token_utils.py`
- Deterministic evaluation harness:
  - `src/eval/harness.py`
- Runnable entry points:
  - `scripts/run_baseline.py`
  - `scripts/run_eval.py`

Run commands:

```powershell
python scripts/run_baseline.py
python scripts/run_baseline.py --backend openrouter --model qwen/qwen3-14b --search-provider duckduckgo
python scripts/run_eval.py
```

Live run requirements:
- `OPENROUTER_API_KEY` for hosted Qwen inference via OpenRouter
- `SERPER_API_KEY` only if using `search-provider serper`
- Replace `qwen/qwen3-14b` with the exact OpenRouter Qwen model ID you plan to use if needed

