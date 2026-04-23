# Repository Guidelines

## Project Structure & Module Organization

This is a Python research workspace for multi-agent LLM orchestration. Root runtime code lives in `src/`: orchestration in `src/orchestrator/`, evaluation in `src/eval/`, critique logic in `src/critique/`, and utilities in `src/tools/`. Executable entry points are in `scripts/`. Configs live under `configs/`, and prompts are in `prompts/roles/` and `prompts/protocol/`.

`arch-1/` and `arch-2/` are separate architecture workspaces. Each keeps implementation code in `implementation/committee_llm/`, configs in `implementation/configs/`, benchmark inputs and run artifacts in `evals/`, and notes in `docs/`. Architecture 2 tests are in `arch-2/tests/`.

## Build, Test, and Development Commands

- `python scripts/run_baseline.py`: run the root baseline using `configs/model/runtime.yaml`.
- `python scripts/run_baseline.py --backend openrouter --model qwen/qwen3-14b --search-provider duckduckgo`: run with provider overrides.
- `python scripts/run_eval.py`: run evaluation and write a JSON report.
- `cd arch-2/implementation && python3.11 -m committee_llm.train_controller ...`: train the Architecture 2 controller; see `ARCH2_START_HERE.md`.
- `cd arch-2 && PYTHONPATH=implementation pytest tests`: run Architecture 2 unit tests.

Install Architecture 2 dependencies with `python3.11 -m pip install -r arch-2/implementation/requirements.txt`. Architecture 1 currently has no third-party requirements listed.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code, four-space indentation, type hints for public functions, and `from __future__ import annotations` in new modules when useful. Follow existing `snake_case` naming for modules, functions, variables, config keys, and run directories. Keep prompt files as Markdown, for example `captain_plan.md` or `critic.md`.

## Testing Guidelines

Prefer focused pytest tests for controller, orchestration, reward, and parsing behavior. Name test files `test_*.py` and test functions `test_*`. Keep tests deterministic: use small tensors, local fixture data, and no live provider calls. For changes requiring API credentials, document the command and provider used.

## Commit & Pull Request Guidelines

Recent commits use short summaries such as `Add Modal Qwen eval setup and smoke results` or concise research notes like `arch 2`. Keep the first line under roughly 72 characters and mention the affected architecture or subsystem when helpful.

Pull requests should describe the goal, list changed areas such as `src/orchestrator` or `arch-2/implementation`, include commands run, and link benchmark reports or issue context. Include screenshots only for generated plots or visual reports.

## Security & Configuration Tips

Live runs may require `OPENROUTER_API_KEY`, `EXA_API_KEY`, or `SERPER_API_KEY`. Set them in the shell or notebook runtime, not in tracked files. Check generated logs and JSON reports for prompt, dataset, or credential leakage before committing.
