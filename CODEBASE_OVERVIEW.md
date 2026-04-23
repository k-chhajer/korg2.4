# Codebase Overview

Generated on 2026-04-16 from the repository state at `/Users/luthiraa/Documents/korg2.4`.

This repository is a research and implementation workspace for a Grok 4.2-style native multi-agent reasoning system. It contains two related but distinct code paths:

1. The root `src/` runtime, which implements the current Milestone 1 four-role orchestration system with live OpenRouter inference and routed web search.
2. The `arch-1/` workspace, which implements a fixed five-stage Architecture 1 committee runner, matched baselines, benchmark conversion, benchmark execution, report generation, Modal/AWS deployment helpers, and accumulated evaluation artifacts.

The repo is on branch `luthira-progress`. At the time this document was written, the working tree already had uncommitted changes in several files plus untracked local additions, so this document describes the current filesystem state rather than a clean committed snapshot.

## High-Level Purpose

The research idea is to approximate a native multi-agent inference-time architecture with specialized roles collaborating around one user query. The target role set in the root runtime is:

- `captain`: decomposes the task, coordinates roles, resolves conflicts, and synthesizes the final answer.
- `harper`: performs web research, factual grounding, query planning, and evidence accounting.
- `benjamin`: performs rigorous logic, code, math, and failure-mode analysis.
- `lucas`: reviews Harper and Benjamin, critiques gaps, and improves communication quality.

The longer-term research plan in `implementation_plan.md` is staged:

- Milestone 0: freeze protocol, reproducibility, eval registry, seed policy.
- Milestone 1: prompt-only native multi-agent baseline.
- Milestone 2: synthetic trace generation and SFT warm start.
- Milestone 3: role-conditioned rewards and verifiers.
- Milestone 4: joint multi-agent RL with AT-GRPO-style training.
- Milestone 5: optional role-specialized LoRA branch.
- Milestone 6: full benchmarking, ablations, and publication package.

The active implemented milestone is Milestone 1. The current runtime is prompt-only and live-provider backed. The repository does not currently ship a mock backend.

## Top-Level Layout

```text
.
├── README.md
├── description.md
├── implementation_plan.md
├── MILESTONE_0_1_STATUS.md
├── CODEBASE_OVERVIEW.md
├── configs/
├── data/
├── prompts/
├── scripts/
├── src/
└── arch-1/
```

### Top-Level Documents

- `README.md`: current user-facing quickstart for the Milestone 1 runtime. It lists the role graph, usage commands, required environment variables, and default runtime config.
- `description.md`: long-form project concept and research rationale. It describes the speculative Grok 4.2-like architecture, the Qwen3/verl/AT-GRPO plan, synthetic trace generation, role rewards, structured critique, and benchmark goals.
- `implementation_plan.md`: structured milestone plan for the root multi-agent project.
- `MILESTONE_0_1_STATUS.md`: concise status note for implemented Milestone 0 and Milestone 1 pieces.
- `LICENSE`: repository license file.

## Root Runtime: `src/`

The root runtime is a Python package for the four-role `captain` / `harper` / `benjamin` / `lucas` workflow.

### Package Layout

```text
src/
├── __init__.py
├── orchestrator/
│   ├── __init__.py
│   ├── backend.py
│   ├── pipeline.py
│   └── types.py
├── critique/
│   ├── __init__.py
│   ├── harper_validator.py
│   └── validator.py
├── eval/
│   ├── __init__.py
│   └── harness.py
├── tools/
│   ├── __init__.py
│   ├── json_utils.py
│   ├── run_logger.py
│   ├── token_utils.py
│   └── web_search.py
├── rewards/
│   └── __init__.py
└── training/
    └── __init__.py
```

The `rewards/` and `training/` packages are placeholders for later reward-modeling and training work. The implemented runtime lives in `orchestrator/`, `critique/`, `eval/`, and `tools/`.

### `src/orchestrator/types.py`

This file defines the shared dataclasses used by the root runtime:

- `ProtocolConfig`: loaded from `configs/protocol/protocol.yaml`. Despite the `.yaml` extension, the file is JSON text and is parsed with `json.loads`.
- `RuntimeConfig`: loaded from `configs/model/runtime.yaml`, also JSON text. It stores backend, search, and generation settings and supports CLI overrides for backend provider, model, and search provider.
- `StageEvent`: per-stage telemetry with stage, role, latency, token count, retries, acceptance status, and errors.
- `SearchResult`: normalized search result with title, URL, snippet, source/provider metadata, publication date, summary, text, highlights, and assigned source ID.
- `SearchTrace`: one search query execution and its results.
- `SearchRequest`: normalized search request including query, mode, result count, domains, location, and language.
- `RunRecord`: full output record for one orchestrator run, including plan, role outputs, critiques, final answer, search traces, events, and metrics.

### `src/orchestrator/backend.py`

This file defines the inference backend abstraction.

- `Backend`: protocol with a single `generate(...) -> str` method.
- `OpenRouterBackend`: OpenRouter chat-completions client implemented with `urllib.request`, not the OpenAI SDK.
- `build_backend`: selects backend implementation from `RuntimeConfig`.

The only supported provider right now is `openrouter`. Live runs require `OPENROUTER_API_KEY`. The default model in config is `qwen/qwen3-14b`, but `scripts/run_baseline.py` and `scripts/run_eval.py` can override it with `--model`.

### `src/orchestrator/pipeline.py`

This is the main root-runtime orchestrator. The central class is `MultiAgentOrchestrator`.

The run graph is:

1. `captain_plan`
   - Loads `prompts/roles/captain_plan.md`.
   - Asks Captain to return strict JSON with overall strategy, role tasks, synthesis checks, and deliverable shape.
   - Falls back to `_default_captain_plan` if JSON parsing fails.

2. `harper_search_plan`
   - Loads `prompts/roles/harper_search_plan.md`.
   - Asks Harper for a strict JSON search plan with mode, queries, category, include/exclude domains, and notes.
   - Falls back to default implementation-oriented queries if parsing fails.

3. `harper_search`
   - Executes each planned query through the configured search tool.
   - Builds `SearchTrace` records for successful and failed searches.
   - Stores per-query telemetry in `StageEvent`.

4. Evidence ledger construction
   - `_build_evidence_ledger` deduplicates results by URL.
   - Assigns stable `source_id` values.
   - Merges snippets, summaries, full text, and highlights where available.

5. `harper_research`
   - Loads `prompts/roles/harper.md`.
   - Passes the captain plan, search plan, serialized search traces, evidence ledger, and user query.
   - Requires Harper to return strict JSON with `answer_summary`, `claims`, `source_digest`, and `open_questions`.
   - Validates with `validate_harper_output`.
   - Retries up to `retry.max_schema_retries`.
   - Falls back to `_default_harper_output` if schema validation never passes.

6. `benjamin_reasoning`
   - Loads `prompts/roles/benjamin.md`.
   - Asks Benjamin for logic, assumptions, failure modes, and checks.

7. `lucas_review`
   - Loads `prompts/roles/lucas.md`.
   - Gives Lucas the captain plan, Harper output, Benjamin output, and user query.
   - Asks for critique, user impact, communication advice, and recommendation.

8. `lucas_critique`
   - Loads `prompts/roles/critique.md` and `prompts/protocol/critique_schema.md`.
   - Asks Lucas for one strict JSON critique with `challenge_type`, `target`, and `correction`.
   - Validates with `validate_critique`.
   - Retries once by default.
   - Falls back to a conservative `missing_consideration` critique if validation fails.
   - Enforces total critique budget through `validate_total_critique_budget`; corrections are truncated if needed.

9. `captain_synthesis`
   - Loads `prompts/roles/captain_synthesis.md`.
   - Gives Captain the plan, search plan, search evidence, evidence ledger, role outputs, accepted critiques, and original query.
   - Produces final answer.

At the end, the orchestrator returns a `RunRecord` with:

- `captain_plan`
- `role_outputs`
- `critiques`
- `final_answer`
- `search_traces`
- `events`
- `metrics`

Current root metrics are simple:

- `total_tokens`: sum of approximate whitespace token counts across events.
- `total_latency_ms`: sum of stage latencies.
- `critique_acceptance_rate`: accepted critique count divided by critique count.

### `src/critique/validator.py`

Validates the short structured critique object. It requires exact JSON with:

- `challenge_type`
- `target`
- `correction`

It rejects:

- invalid JSON
- missing keys
- extra keys
- unsupported challenge type
- target longer than configured token limit
- correction longer than configured token limit
- total critique longer than configured token limit

Allowed challenge types are configured as:

- `factual`
- `logical`
- `missing_consideration`
- `alternative_approach`

### `src/critique/harper_validator.py`

Validates Harper's evidence-grounded JSON output. It enforces:

- output token budget
- parseable JSON object
- required top-level keys
- non-empty `answer_summary`
- bounded `claims`
- bounded `open_questions`
- source digest presence
- claim schema fields: `claim`, `support`, `confidence`, `sources`, `notes`
- support values: `supported`, `partial`, `uncertain`, `contradicted`
- confidence in `[0.0, 1.0]`
- source IDs must be integers and must exist in the evidence ledger
- cited sources must appear in `source_digest`

It normalizes the payload only if there are no validation errors.

### `src/tools/web_search.py`

This file implements all root-runtime search tooling. It uses standard-library HTTP, no third-party search SDK.

Supported providers:

- `DuckDuckGoSearchTool`
  - Scrapes DuckDuckGo HTML results.
  - Requires no API key.
  - Used as a best-effort fallback.

- `SerperSearchTool`
  - Calls `https://google.serper.dev/search`.
  - Requires `SERPER_API_KEY`.
  - Used for `fast_lookup` by default.

- `ExaSearchTool`
  - Calls Exa search and optional content endpoints.
  - Requires `EXA_API_KEY`.
  - Used for `standard_research` and `deep_research` by default.
  - Can fetch text, summaries, and highlights depending on mode and config.

- `SearchRouter`
  - Routes modes to providers.
  - Can fall back to a fallback provider if primary search fails.
  - Raises if configured providers are missing credentials.

- `FixedProviderSearchTool`
  - Wraps one provider when `search.provider` is explicitly set.

Search modes:

- `fast_lookup`: maps to Exa type `fast`; default provider is Serper.
- `standard_research`: maps to Exa type `auto`; default provider is Exa.
- `deep_research`: maps to Exa type `deep`; default provider is Exa.

### `src/tools/json_utils.py`

JSON parsing helpers:

- strips Markdown code fences if present
- tries `json.loads`
- if that fails, scans for the first `{` or `[` and tries `JSONDecoder.raw_decode`
- raises `ValueError("no_json_payload_found")` if no parseable JSON exists

This is used for robust parsing of model outputs that may include surrounding text.

### `src/tools/token_utils.py`

Defines `count_tokens_approx(text)`, a cheap whitespace token proxy used in local deterministic metrics and validation. It is not a tokenizer-specific count.

### `src/tools/run_logger.py`

Defines:

- `ensure_dir(path)`
- `write_json(path, payload)`

Used by scripts to create output directories and write JSON traces or reports.

### `src/eval/harness.py`

Implements the lightweight root eval harness:

- `load_jsonl`: reads JSONL tasks.
- `keyword_score`: fraction of expected keywords present in final answer.
- `role_distinctiveness`: pairwise Jaccard-distance-like token-set distinctiveness across role outputs.
- `run_eval`: runs the orchestrator on each task and aggregates:
  - mean keyword score
  - mean role distinctiveness
  - mean critique acceptance rate

The root eval harness is intentionally simple and is not the same as the more developed `arch-1` benchmark harness.

## Root Configs

The root config files use `.yaml` extensions but contain JSON syntax.

### `configs/protocol/protocol.yaml`

Protocol version: `0.1.0`.

Roles:

- `captain`
- `harper`
- `benjamin`
- `lucas`

Turn order:

- `captain_plan`
- `harper_search_plan`
- `harper_search`
- `harper_research`
- `benjamin_reasoning`
- `lucas_review`
- `lucas_critique`
- `captain_synthesis`

Token budgets:

- Captain plan: `256`
- Harper search plan: `96`
- Harper output: `420`
- Benjamin output: `420`
- Lucas output: `320`
- Generic role output fallback: `320`
- Critique: `60`
- Captain synthesis: `640`
- Total critique budget: `180`

Retry policy:

- `max_schema_retries`: `1`

Seed policy:

- `global_seed`: `42`
- `task_seed_offset`: `1000`

### `configs/model/runtime.yaml`

Default backend:

- provider: `openrouter`
- model: `qwen/qwen3-14b`
- endpoint: `https://openrouter.ai/api/v1/chat/completions`
- timeout: `60` seconds

Default search:

- provider: `router`
- default mode: `fast_lookup`
- max queries: `3`
- max results: `5`
- location: `US`
- language: `en`
- routing:
  - `fast_lookup`: `serper`
  - `standard_research`: `exa`
  - `deep_research`: `exa`
  - fallback: `duckduckgo`

Generation temperatures:

- default: `0.2`
- captain plan: `0.1`
- Harper search plan: `0.0`
- Harper research: `0.2`
- Benjamin reasoning: `0.1`
- Lucas review: `0.4`
- Lucas critique: `0.1`
- Captain synthesis: `0.2`

### `configs/eval/eval_config.yaml`

Defines:

- dataset: `data/processed/eval_suite.jsonl`
- report directory: `reports/benchmark_runs`
- max tasks: `200`

### `configs/experiments/registry.yaml`

Defines experiment IDs and statuses:

- `A0`: single-agent base, planned.
- `A1`: prompt-only multi-agent, active.
- `A2`: SFT multi-agent, planned.
- `A3`: joint RL shared policy, planned.
- `A4`: joint RL role LoRA, planned.
- `A5`: independent role training, planned.

## Root Prompts

Root prompt files live under `prompts/`.

### Protocol Prompts

- `prompts/protocol/system.md`
  - Defines the four roles.
  - Instructs role adherence, concise structure, Harper grounding, strict Harper JSON, Benjamin assumptions, Lucas review, JSON critique, and Captain contradiction resolution.

- `prompts/protocol/critique_schema.md`
  - Defines the exact critique JSON schema.
  - Enforces no Markdown and no extra keys.
  - Sets target and correction length constraints.

### Role Prompts

- `prompts/roles/captain_plan.md`
  - Captain decomposes the query and returns strict JSON with role-specific subtask assignments.

- `prompts/roles/harper_search_plan.md`
  - Harper chooses search mode and up to three high-signal queries.

- `prompts/roles/harper.md`
  - Harper consumes search findings and evidence ledger.
  - Returns strict JSON with grounded claims, confidence, support status, source digest, and open questions.

- `prompts/roles/benjamin.md`
  - Benjamin outputs logic, assumptions, failure modes, and checks.

- `prompts/roles/lucas.md`
  - Lucas reviews Harper and Benjamin, then outputs critique, user impact, communication advice, and recommendation.

- `prompts/roles/critique.md`
  - Asks one role, currently Lucas in the pipeline, for one concise schema-valid critique against current drafts.

- `prompts/roles/captain_synthesis.md`
  - Captain integrates all role outputs and accepted critique.
  - Requires short `facts`, `logic`, and `alternative` sections and an actionable recommendation.

## Root Data

### `data/processed/eval_suite.jsonl`

Contains eight starter eval tasks:

- legacy Python monolith migration
- database sharding approaches
- recursive vs iterative algorithm analysis
- fintech onboarding flow
- API backward compatibility audit
- flaky CI debugging
- low-latency model evaluation
- hallucination reduction for retrieval assistant

Each task expects the final answer to include `facts`, `logic`, and `alternative`, matching the Captain synthesis prompt.

## Root Scripts

### `scripts/run_baseline.py`

Runs one prompt through the root Milestone 1 pipeline.

Default query:

```text
Create an implementation plan for a multi-agent reasoning system.
```

CLI options:

- `--query`
- `--task-id`
- `--seed`
- `--runtime-config`
- `--backend`
- `--model`
- `--search-provider`

It writes a run JSON file under `reports/benchmark_runs/` and prints runtime info, metrics, and final answer.

Example commands from `README.md`:

```powershell
python scripts/run_baseline.py
python scripts/run_baseline.py --search-provider router
python scripts/run_baseline.py --backend openrouter --model qwen/qwen3-14b --search-provider duckduckgo
```

### `scripts/run_eval.py`

Runs the root eval harness over `data/processed/eval_suite.jsonl`.

It loads:

- `configs/eval/eval_config.yaml`
- `configs/protocol/protocol.yaml`
- `configs/model/runtime.yaml`

It writes an eval report JSON under `reports/benchmark_runs/` and prints aggregate metrics.

Example:

```powershell
python scripts/run_eval.py
```

## Root Runtime Environment Variables

Live root runs can require:

- `OPENROUTER_API_KEY`: required for OpenRouter backend.
- `EXA_API_KEY`: required for Exa search.
- `SERPER_API_KEY`: required for Serper search.

Notes:

- `duckduckgo` can run without a key but is less reliable.
- The default `router` provider requires Serper and Exa credentials because its configured primary routes include both.
- If you want no-key search, use `--search-provider duckduckgo`.

## Architecture 1 Workspace: `arch-1/`

`arch-1/` is a separate, larger workspace for a fixed committee control condition. It has its own implementation package, configs, docs, data, test suite, run artifacts, and deployment helpers.

Top-level structure:

```text
arch-1/
├── aws/
├── docs/
├── evals/
└── implementation/
```

The Architecture 1 committee is not the same as the root `captain` / `harper` / `benjamin` / `lucas` runtime. It uses:

- `coordinator_plan`
- `researcher`
- `analyst`
- `critic`
- `coordinator_finalize`

The fixed topology is:

```text
coordinator -> researcher -> analyst -> critic -> coordinator
```

The docs explicitly state that this is an Architecture 1 control condition, not the primary novelty claim.

## `arch-1/implementation/committee_llm`

This is the main package for Architecture 1.

```text
arch-1/implementation/committee_llm/
├── __init__.py
├── __main__.py
├── baselines.py
├── benchmark.py
├── benchmark_data.py
├── benchmark_report.py
├── client.py
├── cli.py
├── config.py
├── evaluation.py
├── metrics.py
├── orchestrator.py
└── tasks.py
```

### `config.py`

Defines the config model and fixed Architecture 1 constraints.

Important constants:

- `ROLE_SEQUENCE`
  - `coordinator_plan`
  - `researcher`
  - `analyst`
  - `critic`
  - `coordinator_finalize`

- `BASE_ROLE_ORDER`
  - `coordinator`
  - `researcher`
  - `analyst`
  - `critic`
  - `coordinator`

- `DEFAULT_OUTPUT_SCHEMAS`
  - Each stage has required section headers.
  - Example: `coordinator_plan` requires `Problem Restatement`, `Success Criteria`, `Plan`, `Risks and Unknowns`, `Research Brief`.

- `DEFAULT_LOCAL_CONTEXT_VIEWS`
  - Defines exactly which task and prior-stage blocks each role sees.
  - Later roles receive progressively more prior stage outputs.

Dataclasses:

- `ProviderConfig`
- `DefaultRoleSettings`
- `RoleConfig`
- `ArchitectureSpec`
- `ExperimentConfig`

`ExperimentConfig.load(...)` parses a JSON config and enforces the exact fixed role sequence. It also resolves relative prompt paths and provides role settings, role schemas, context views, system prompts, and paper-claim blocker checks.

Paper-valid Architecture 1 requires:

- `architecture.specialization_source = post_trained_role_specialists`
- each role has a checkpoint identifier
- role checkpoints are not reused in a way that violates the control definition

Prompt-only scaffold configs are explicitly flagged as not paper-valid.

### `client.py`

Defines `OpenAICompatibleChatClient`, a minimal standard-library client for OpenAI-compatible `/v1/chat/completions` endpoints.

It supports:

- arbitrary `api_base`
- API key from configured environment variable
- timeout
- model
- messages
- temperature
- max tokens
- optional extra request body

It returns a `ChatResult` with:

- model
- content
- elapsed seconds
- usage dict
- raw response

This allows Architecture 1 to run against Ollama, vLLM, llama.cpp `llama-server`, LM Studio, Modal vLLM endpoints, or any compatible server.

### `orchestrator.py`

Defines `CommitteeRunner`, the fixed Architecture 1 runner.

Run flow:

1. Convert a prompt or structured dict into `TaskSpec`.
2. For each stage in `config.architecture.stage_sequence`:
   - resolve role model/settings
   - load role system prompt
   - build role-specific user prompt from task blocks and prior stage outputs
   - call the OpenAI-compatible chat client
   - validate that required section headers are present
   - record stage response, prompt, schema validation, usage, elapsed time, model, temperature, local context view, output schema, role specialization, and checkpoint name
3. Extract final answer from the `Final Answer:` section of `coordinator_finalize`.
4. Build trace summary:
   - stage count
   - model call count
   - per-role elapsed time
   - per-role usage
   - aggregate usage
5. Evaluate final answer through `evaluation.py`.
6. Compute behavior metrics:
   - decomposition rate
   - evidence grounding rate
   - revision after critique rate
   - premature finalization rate
   - unsupported claim rate
7. Return a full trace dict.

The trace includes the architecture spec and paper-claim blockers, which is important because many checked-in configs are prompt scaffolds rather than true post-trained role-specialist checkpoints.

### `tasks.py`

Defines `TaskSpec`, the structured task format used by the Architecture 1 runner and benchmark tools.

Supported task fields:

- `prompt`
- `id`
- `domain`
- `benchmark_name`
- `benchmark_split`
- `task_type`
- `context`
- `objectives`
- `constraints`
- `deliverable`
- `evaluation_criteria`
- `references`
- `reference_answers`
- `choices`
- `correct_choice`
- `checks`
- `metadata`

Supported task types:

- `open_qa`
- `multiple_choice`

Supported deterministic checks:

- `must_contain`
- `must_not_contain`
- `min_words`
- `max_words`
- `min_chars`
- `max_chars`

Multiple-choice tasks use unique labels, normally `A` through `D`, and enforce that `correct_choice` matches one label.

### `evaluation.py`

Evaluates final answers for benchmark and deterministic checks.

For open QA:

- normalizes answers by lowercasing, removing punctuation, articles, and extra whitespace
- computes exact match
- computes token F1
- scores best candidate over multiple reference answers
- extracts candidate answer spans from explanatory outputs so leading short-answer phrases can score correctly

For multiple choice:

- extracts labels from patterns like `Answer: C`, `option C`, or leading `C.`
- can match choice text inside the normalized answer
- computes `choice_accuracy`

For HotpotQA-style metadata:

- computes supporting-fact title recall
- computes supporting-fact title precision
- computes supporting-fact context coverage

It also reports deterministic check pass/fail, answer word count, and answer character count.

### `baselines.py`

Implements matched single-model baseline runners:

- `SingleShotRunner`
  - One model call.
  - Solves directly in one response.
  - Asks for shortest direct answer and brief justification.

- `SingleModelMultiTurnRunner`
  - Three model calls in one conversation:
    - plan
    - evidence
    - final answer

- `SingleModelStructuredRunner`
  - One model call.
  - Forces five headers:
    - Decomposition
    - Evidence
    - Tentative Answer
    - Self-Critique
    - Final Answer

All baseline traces use the same `TaskSpec` and evaluation machinery as the committee runner.

### `benchmark.py`

Batch benchmark runner for Architecture 1 and baselines.

Systems:

- `single_shot`
- `single_multiturn`
- `single_structured`
- `committee`
- `all`

Features:

- loads JSONL tasks
- runs selected systems
- writes per-task JSON traces under `outdir/<system>/<task_id>.json`
- supports `--limit`
- supports `--continue-on-error`
- supports `--resume` to reuse existing per-task JSON outputs
- checkpoint-writes summary after each task

The summary includes:

- completed and failed counts
- wall-clock time
- mean run time
- tasks per second
- aggregate usage
- per-role usage
- model call counts
- overall reference metrics
- per-benchmark reference metrics
- behavior metrics
- committee-vs-baseline ratios and metric deltas
- failures

### `benchmark_data.py`

Converts external benchmark formats into the internal `TaskSpec` JSONL format.

Supported converters:

- `hotpotqa`
  - Input: HotpotQA JSON array.
  - Output: open-QA tasks with provided context, references, gold answer, supporting-fact metadata.

- `musique`
  - Input: JSON or JSONL.
  - Output: open-QA tasks with paragraphs, aliases, and references.

- `gpqa`
  - Input: CSV.
  - Output: multiple-choice tasks.
  - Shuffles correct and incorrect answers deterministically using question ID or question text.

### `benchmark_report.py`

Renders a benchmark `summary.json` into Markdown.

It produces sections for:

- verification and paper-claim blockers
- system overview
- overall metrics
- per-benchmark metrics
- behavior metrics
- comparisons
- failures

### `cli.py` and `__main__.py`

`cli.py` provides a one-off and batch runner for the committee system. It can run:

- a single prompt with `--prompt`
- a prompt file with `--prompt-file`
- a structured task file with `--task`
- a JSONL batch with `--tasks`

It writes JSON traces and batch summaries. `__main__.py` delegates to `committee_llm.cli.main`, so it can be invoked with `python -m committee_llm`.

### `metrics.py`

Small numeric aggregation helpers:

- recursively merges numeric dicts
- aggregates usage dicts across stages or runs

## `arch-1/implementation/prompts`

Architecture 1 role prompts live here:

- `coordinator_plan.md`
- `researcher.md`
- `analyst.md`
- `critic.md`
- `coordinator_finalize.md`

These are role-specific system prompts for the five-stage committee. The runtime also injects per-stage task blocks and required section-header instructions from config.

## `arch-1/implementation/configs`

This directory contains multiple runnable and template configs for Architecture 1.

Important examples:

- `qwen3_8b_ollama_eval.json`
  - Local Ollama config.
  - API base: `http://127.0.0.1:11434/v1`.
  - Model: `qwen3:8b`.
  - Prompt scaffold only.

- `qwen3_8b_modal_t4_awq_eval.json`
  - Modal-hosted Qwen3-8B-AWQ vLLM endpoint.
  - API base points to `tryjulieai--korg-arch1-qwen3-t4-awq-vllm-serve.modal.run`.
  - Uses `MODAL_API_TOKEN`.
  - Prompt scaffold only.
  - Enables Qwen3 thinking through `chat_template_kwargs.enable_thinking` on selected coordinator stages.

- `qwen3_arch1_posttrained_template.json`
  - Paper-grade template.
  - Declares `specialization_source = post_trained_role_specialists`.
  - Uses distinct role model/checkpoint names such as `korg/qwen3-8b-researcher-sft`.
  - This is a template; the actual role-specialized checkpoints are not shipped.

Other configs cover Qwen3 8B/30B committee and eval variants, Modal, Colab T4, and thinking-mode runs.

## `arch-1/implementation/modal_vllm_qwen3*.py`

Modal deployment scripts for serving OpenAI-compatible vLLM endpoints.

### `modal_vllm_qwen3.py`

- App: `korg-arch1-qwen3-vllm`
- Model: `Qwen/Qwen3-8B`
- GPU: `L4`
- vLLM: `0.10.2`
- Transformers: `4.56.1`
- Hugging Face transfer enabled
- Serves on port `8000`
- Uses Modal volume `korg-arch1-qwen3-cache`

### `modal_vllm_qwen3_t4_awq.py`

- App: `korg-arch1-qwen3-t4-awq-vllm`
- Model: `Qwen/Qwen3-8B-AWQ`
- GPU: `T4`
- AWQ quantization enabled
- Max model length: `8192`
- Max sequences: `16`
- Uses Modal volume `korg-arch1-qwen3-t4-awq-cache`

Both scripts run `python -m vllm.entrypoints.openai.api_server` inside Modal and expose the OpenAI-compatible server as a web endpoint.

## `arch-1/docs`

Docs in this directory explain the Architecture 1 boundary, benchmark plan, local framework interpretation, remote eval plan, and current results.

Important files:

- `README.md`
  - Orients the `arch-1` workspace.
  - Points to implementation, evals, docs, and key entry points.

- `ARCHITECTURE_1.md`
  - Defines Architecture 1 as four separate instances of the same base model in a fixed coordinator/researcher/analyst/critic/coordinator topology.
  - Explains what code enforces now.
  - Separates paper-valid post-trained specialists from prompt-only scaffolds.

- `BENCHMARKS.md`
  - Defines the benchmark protocol.
  - Main benchmark: HotpotQA, 100 examples.
  - Secondary benchmark: FRAMES only if retrieval is active.
  - Stress benchmark: custom 50 examples.
  - Systems: `single_shot`, `single_multiturn`, `single_structured`, `committee`.
  - Metrics include answer accuracy/F1, supporting-fact proxies, token usage, model calls, latency, and behavior metrics.
  - Required ablations include no critic, no researcher, no analyst, two order swaps, and budget-matched single-model baseline.

- `BENCHMARK_RESULTS.md`
  - Current HotpotQA results summary.
  - Uses first 75 examples from `hotpotqa_dev.jsonl`.
  - Hardware: AWS `g4dn.xlarge`, Qwen3-8B-AWQ through local vLLM.
  - Committee completed `73/75`; single-shot baselines completed `75/75`.
  - Committee answer F1 on completed traces: `0.5774`.
  - Conservative full-set committee answer F1: `0.5620`.
  - Single-shot answer F1: `0.0162`.
  - Committee is slower and more token-intensive but materially better on this run.

- `LOCAL_COMMITTEE_FRAMEWORK.md`
  - Explains that `arch-1` is already close to a reusable local committee framework.
  - Documents how to run with Ollama, llama.cpp, or vLLM-style endpoints.
  - Summarizes HotpotQA and GPQA improvement factors and caveats.

- `REMOTE_EVAL_PLAN.md`
  - Modal/T4-oriented remote eval sequence.
  - Recommends endpoint verification, small smoke benchmark, main HotpotQA, custom stress, FRAMES only if retrieval is active, then ablations.
  - Provides budget guidance.

- `COLAB_T4_GPQA.md` and `COLAB_T4_NOTEBOOK_CELLS.md`
  - Colab-oriented GPQA/T4 evaluation notes and notebook-cell instructions.

- `IMPLEMENTATION_NOTE.md` and `summary.md`
  - Additional implementation and research notes.

## `arch-1/evals`

Architecture 1 evaluation data, tests, reports, and run outputs live here.

```text
arch-1/evals/
├── data/
├── reports/
├── runs/
└── tests/
```

### `arch-1/evals/data`

Important files:

- `hotpot_dev_distractor_v1.json`
  - Raw HotpotQA dev distractor data.

- `benchmarks/hotpotqa_dev.jsonl`
  - Converted HotpotQA benchmark tasks.

- `benchmarks/hotpotqa_dev_20.jsonl`
  - Smaller converted HotpotQA subset.

- `benchmarks/gpqa_diamond.csv`
  - GPQA Diamond source CSV.

- `benchmarks/gpqa_diamond.jsonl`
  - Converted GPQA Diamond task JSONL.

- `sample_tasks.jsonl`
  - Small sample JSONL task set.

- `sample_research_task.json`
  - Single structured sample research task.

### `arch-1/evals/runs`

This directory contains accumulated benchmark and probe runs. Examples:

- `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414`
- `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b`
- `hotpotqa_modal_100`
- `hotpotqa_modal_smoke_2`
- `hotpotqa_qwen3_8b_ollama_20`
- `hotpotqa_qwen3_8b_ollama_20_smoke`
- `hotpotqa_qwen3_8b_ollama_20_smoke_v2`
- `gpqa_diamond_modal_committee_198_20260405`
- `gpqa_diamond_modal_committee_5pilot_20260403`
- `gpqa_modal_t4_awq_thinking_198_20260405`
- several GPQA probe and retry runs

Many run directories contain:

- per-task trace JSON files
- `summary.json`
- logs such as `benchmark.log` and `vllm.log`
- retry task JSONL files
- shard summaries

The run artifacts are data/results, not source code, but they are important for research provenance.

### `arch-1/evals/reports`

Contains rendered reports and plotting helpers:

- `benchmark_comparison_summary.md`
- `gpqa_diamond_modal_committee_198_20260405.md`
- `graphs/eval_graph_data.csv`
- `plot_eval_graphs.py`

### `arch-1/evals/tests`

Unit tests for the Architecture 1 package:

- `test_orchestrator.py`
  - Tests committee trace structure, usage aggregation, paper-claim blockers, structured task handling, evaluation, schema validation, and role metadata.

- `test_benchmark.py`
  - Tests multi-system benchmark summary generation using fake runners.
  - Verifies comparison deltas, token ratios, and behavior metric aggregation.

- `test_benchmark_data.py`
  - Tests HotpotQA and GPQA conversion into internal JSONL.

- `test_tasks.py`
  - Tests `TaskSpec` parsing, deterministic checks, open-QA metrics, multiple-choice metrics, and short-answer extraction.

- `test_cli.py`
  - Tests CLI behavior.

- `test_benchmark_report.py`
  - Tests Markdown report rendering.

Recommended command from docs:

```powershell
cd arch-1
$env:PYTHONPATH = "implementation"
python -m unittest discover -s evals\tests
```

On macOS/Linux shell, the equivalent would be:

```bash
cd arch-1
PYTHONPATH=implementation python -m unittest discover -s evals/tests
```

## `arch-1/aws`

AWS setup and run scripts for GPU benchmarking.

Important files:

- `SETUP.md`
  - Documents AWS EC2 setup as of 2026-04-05.
  - Includes key pair names, security group IDs, launch template IDs, AMIs, instance type, region, subnet, and blocker notes.

- `ec2_user_data.sh`
  - EC2 bootstrap script.

- `launch_template_data.json`
  - Launch template data for `us-east-2`.

- `launch_template_data_use1.json`
  - Launch template data for `us-east-1`.

- `debug_vllm_use1.sh`
  - Debug helper for vLLM in `us-east-1`.

- `run_gpqa_aws_use1.sh`
  - GPQA AWS run helper.

- `start_hotpot_retry4b.sh`
  - HotpotQA retry/committee run helper.

- `start_hotpot_single_shot_thinking_20260415.sh`
  - Single-shot thinking-mode HotpotQA run helper.

Private key files are present in this directory:

- `arch1-g4dn-20260405.pem`
- `arch1-g4dn-use1-20260405.pem`

These are sensitive local files. They should not be committed or shared.

## Generated and Cache Files

The repository currently contains multiple `__pycache__/` directories under:

- `scripts/`
- `src/`
- `src/tools/`
- `src/training/`
- `src/rewards/`
- `src/eval/`
- `src/critique/`
- `src/orchestrator/`
- `arch-1/evals/tests/`
- `arch-1/implementation/`
- `arch-1/implementation/committee_llm/`

These are Python bytecode caches and are not source of truth.

## Current Git Working Tree Notes

At documentation time, `git status --short` reported modified files:

- `arch-1/docs/BENCHMARK_RESULTS.md`
- `arch-1/evals/reports/benchmark_comparison_summary.md`
- `arch-1/implementation/committee_llm/baselines.py`
- `arch-1/implementation/committee_llm/config.py`

And untracked files/directories:

- `arch-1/aws/start_hotpot_single_shot_thinking_20260415.sh`
- `arch-1/docs/LOCAL_COMMITTEE_FRAMEWORK.md`
- `arch-1/evals/reports/graphs/`
- `arch-1/evals/reports/plot_eval_graphs.py`
- `arch-1/implementation/configs/qwen3_8b_colab_t4_thinking_eval.json`

This matters because some docs and results may be newer than the last committed state.

## How the Two Systems Relate

The root runtime and `arch-1` overlap conceptually but are not identical.

Root runtime:

- roles: Captain, Harper, Benjamin, Lucas
- includes live web-search routing
- uses strict JSON for Captain plan, Harper findings, and critique
- logs `RunRecord` dataclasses
- has a lightweight keyword/distinctiveness eval harness
- is tied to OpenRouter for generation

Architecture 1:

- roles/stages: coordinator plan, researcher, analyst, critic, coordinator finalize
- uses OpenAI-compatible local/remote endpoints
- does not perform live web retrieval in the current scaffold
- enforces required section headers instead of JSON schemas
- has a richer benchmark harness and baseline comparisons
- has real HotpotQA/GPQA run artifacts and reports

The root runtime is closer to the stated Grok-style role design. `arch-1` is more mature as a benchmarking and local committee execution framework.

## Main Commands

Root baseline:

```bash
python scripts/run_baseline.py
```

Root baseline with DuckDuckGo fallback-only search:

```bash
python scripts/run_baseline.py --search-provider duckduckgo
```

Root eval:

```bash
python scripts/run_eval.py
```

Architecture 1 tests:

```bash
cd arch-1
PYTHONPATH=implementation python -m unittest discover -s evals/tests
```

Architecture 1 local Ollama benchmark example:

```bash
cd arch-1
PYTHONPATH=implementation OPENAI_API_KEY=EMPTY python -m committee_llm.benchmark \
  --config implementation/configs/qwen3_8b_ollama_eval.json \
  --tasks evals/data/benchmarks/hotpotqa_dev_20.jsonl \
  --outdir evals/runs/hotpotqa_arch1_local \
  --system committee \
  --limit 20
```

Architecture 1 report rendering:

```bash
cd arch-1
PYTHONPATH=implementation python -m committee_llm.benchmark_report \
  --summary evals/runs/hotpotqa_arch1_local/summary.json \
  --output docs/BENCHMARK_RESULTS.md
```

Modal deployment examples:

```bash
modal deploy arch-1/implementation/modal_vllm_qwen3.py
modal deploy arch-1/implementation/modal_vllm_qwen3_t4_awq.py
```

## Known Boundaries and Caveats

- The root runtime currently has only an OpenRouter backend. There is no local backend implementation in `src/orchestrator/backend.py`.
- Root config files are named `.yaml` but parsed as JSON.
- Root eval metrics are starter metrics, not paper-grade benchmark metrics.
- Root token counts are approximate whitespace counts, not model tokenizer counts.
- Root search router requires provider credentials for configured primary routes unless using `duckduckgo` directly.
- `arch-1` configs currently include prompt scaffolds. Prompt scaffolds are not paper-valid Architecture 1 specialist checkpoints.
- Paper-valid Architecture 1 requires actual distinct post-trained role-specialist checkpoints.
- `arch-1` has strong benchmark tooling, but current scaffold does not include active retrieval; docs say FRAMES should not be run until retrieval is truly part of the pipeline.
- Sensitive AWS private key files are present locally under `arch-1/aws`.
- Large run artifacts are checked into or present in the workspace; they are useful for provenance but make `rg --files` output large.

## Current Research Signal in Artifacts

The strongest measured result currently documented is HotpotQA first-75:

- Single-shot non-thinking Qwen3-8B-AWQ:
  - completed: `75/75`
  - answer F1: `0.0162`
  - total tokens: `174,866`
  - mean run: `23.0636s`

- Single-shot thinking Qwen3-8B-AWQ:
  - completed: `75/75`
  - answer F1: `0.0162`
  - total tokens: `174,866`
  - mean run: `21.5607s`

- Committee partial thinking:
  - completed: `73/75`
  - completed-trace answer EM: `0.3836`
  - completed-trace answer F1: `0.5774`
  - conservative full-set answer EM: `0.3733`
  - conservative full-set answer F1: `0.5620`
  - total tokens: `1,543,424`
  - mean run: `150.0377s`

Interpretation:

- The committee run is much slower and uses far more tokens.
- It is the only run in the documented HotpotQA slice with materially nonzero answer accuracy.
- The reported F1 factor is large because the single-shot baseline is near zero; absolute gain is more stable.

GPQA notes from `LOCAL_COMMITTEE_FRAMEWORK.md`:

- GPQA Diamond committee completed `163/198` examples before Modal billing cap.
- Completed-trace accuracy: `51.53%`.
- Conservative full-set lower bound: `42.42%`.
- It beats the non-thinking Qwen3-8B official reference of `39.30%` on completed traces and conservative lower bound.
- It does not beat the official thinking Qwen3-8B reference of `62.00%`.

## Practical Next Steps Suggested by the Repo

The docs and current code point toward these next engineering steps:

1. Harden the root runtime with tests comparable to `arch-1/evals/tests`.
2. Add a local OpenAI-compatible backend option to the root runtime, reusing ideas from `arch-1/implementation/committee_llm/client.py`.
3. Package `arch-1` as a cleaner standalone local committee framework with `committee run` and `committee bench` commands.
4. Add budget controls:
   - max total tokens per task
   - max wall time per stage
   - optional early stop on schema failure
5. Add root-runtime benchmark conversion or connect it to `arch-1`'s richer `TaskSpec`/evaluation machinery.
6. Decide whether root roles and Architecture 1 roles should remain separate or be unified behind one configurable runner.
7. Create true role-specialized checkpoints if making paper-valid Architecture 1 claims.
8. Run required ablations once the main benchmark trend is stable.
