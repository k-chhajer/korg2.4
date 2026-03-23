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
- Structured critique validation and retry handling:
  - `src/critique/validator.py`
- Prompt protocol and role templates:
  - `prompts/protocol/*`
  - `prompts/roles/*`
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
python scripts/run_baseline.py --task-id smoke_test --query "Design a starter plan for a multi-agent research system."
python scripts/run_eval.py
```

