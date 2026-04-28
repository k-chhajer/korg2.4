# Architecture 1 Comprehensive Reference

This document is the full engineering reference for `arch-1` in this repository. It consolidates the architecture definition, implementation details, role contracts, deployment setups, evaluation methodology, recorded benchmark runs, results, caveats, and reproduction workflow.

Last updated from the checked-in repository state on `2026-04-28`.

## 1. Executive Summary

Architecture 1 is the explicit external-specialization control condition for the paper. In this repo it is implemented as a fixed five-stage committee pipeline:

`coordinator_plan -> researcher -> analyst -> critic -> coordinator_finalize`

The main implementation status is:

- the code enforces the fixed topology
- each stage has its own prompt, schema, and local context view
- each stage is a separate inference call
- benchmark harnesses compare the committee against single-model baselines
- checked-in runnable configs are scaffold configs using one underlying base model family with prompt specialization
- paper-valid Architecture 1 claims require distinct role-specific post-trained checkpoints, which are represented as a template but are not shipped here

The strongest checked-in benchmark result is the `75`-example HotpotQA committee run on `Qwen/Qwen3-8B-AWQ`, where the committee completed `73/75` tasks and reached:

- `38.36%` answer EM
- `57.74%` answer F1
- `150.04s` mean elapsed time per completed task
- `1,543,424` total tokens

The strongest checked-in GPQA result is the partially completed `198`-example Modal committee run on `Qwen/Qwen3-8B`, where:

- `163/198` examples completed
- completed-trace accuracy was `51.53%`
- conservative full-set lower bound was `42.42%`
- the main blocker was Modal billing-cap `HTTP 429` failures

Important validity boundary:

- these results are useful engineering scaffold benchmarks
- they are not yet paper-valid evidence for a four-checkpoint post-trained Architecture 1 system

## 2. Repository Layout

Architecture 1 lives under [`arch-1`](/Users/luthiraa/Documents/korg2.4/arch-1).

Primary subdirectories:

- [`arch-1/implementation`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation): committee runtime, configs, prompts, deployment entrypoints
- [`arch-1/evals`](/Users/luthiraa/Documents/korg2.4/arch-1/evals): benchmark inputs, tests, run artifacts, reports
- [`arch-1/docs`](/Users/luthiraa/Documents/korg2.4/arch-1/docs): architecture notes, benchmark plans, generated reports
- [`arch-1/aws`](/Users/luthiraa/Documents/korg2.4/arch-1/aws): AWS launch and benchmark scripts for GPU-backed runs

Most important files:

- [`arch-1/implementation/committee_llm/config.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/config.py)
- [`arch-1/implementation/committee_llm/orchestrator.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/orchestrator.py)
- [`arch-1/implementation/committee_llm/baselines.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/baselines.py)
- [`arch-1/implementation/committee_llm/benchmark.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/benchmark.py)
- [`arch-1/implementation/committee_llm/evaluation.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/evaluation.py)
- [`arch-1/implementation/configs/qwen3_arch1_posttrained_template.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_arch1_posttrained_template.json)
- [`arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md)
- [`arch-1/evals/reports/benchmark_comparison_summary.md`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/reports/benchmark_comparison_summary.md)
- [`arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md)

## 3. Architecture Definition

The required definition in this repo is:

- four separate instances of the same base model family
- roles: coordinator, researcher, analyst, critic
- fixed topology: coordinator -> researcher -> analyst -> critic -> coordinator
- separate system prompt per role
- separate role-specific output schema per role
- separate local context view per role
- separate inference call per role
- specialization must come from role-specific post-training for paper-valid claims, not just prompt wording

The enforced internal stage sequence is:

1. `coordinator_plan`
2. `researcher`
3. `analyst`
4. `critic`
5. `coordinator_finalize`

The paper reference role order is:

`coordinator -> researcher -> analyst -> critic -> coordinator`

The implementation rejects configs that alter the stage sequence away from this fixed order.

## 4. What Counts As Valid

### 4.1 Paper-valid Architecture 1

A config is paper-valid only when:

- `architecture.family = architecture_1`
- `architecture.specialization_source = post_trained_role_specialists`
- the stage sequence remains the fixed five-stage route
- the role order remains coordinator -> researcher -> analyst -> critic -> coordinator
- each role declares role metadata
- each role declares a non-empty schema
- each role declares a non-empty local context view
- each role points at a distinct role-specific checkpoint where required by the specialization setting

### 4.2 Scaffold-only Architecture 1

The checked-in runnable configs are scaffold configs when:

- the same underlying Qwen checkpoint is reused for all roles
- role differences come from prompts, temperatures, and token budgets rather than separate post-trained checkpoints
- `paper_claim_blockers()` returns the specialization-source blocker

The canonical blocker message in the current code is:

- `Role specialization source is not set to post_trained_role_specialists, so this config should be treated as a prompt scaffold.`

### 4.3 Template For The Real Paper Version

The paper-grade template config is:

- [`arch-1/implementation/configs/qwen3_arch1_posttrained_template.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_arch1_posttrained_template.json)

It assumes distinct role checkpoints:

- `korg/qwen3-8b-coordinator-sft`
- `korg/qwen3-8b-researcher-sft`
- `korg/qwen3-8b-analyst-sft`
- `korg/qwen3-8b-critic-sft`

The final coordinator stage still uses the coordinator specialization but has its own checkpoint name field:

- `korg/qwen3-8b-coordinator-sft-v2`

## 5. Control Flow

### 5.1 Pipeline Topology

```text
coordinator_plan
  -> researcher
  -> analyst
  -> critic
  -> coordinator_finalize
```

### 5.2 Why The Topology Exists

The design intent is:

- stage 1 decomposes and defines success criteria
- stage 2 surfaces evidence, distinctions, options, and open questions
- stage 3 builds the draft answer
- stage 4 attacks the draft for unsupported or weak reasoning
- stage 5 integrates the valid critique and produces the final output

This architecture deliberately trades compute for structured reasoning pressure and a mandatory critique pass.

## 6. Stage Contracts

### 6.1 Coordinator Plan

Prompt file:

- [`arch-1/implementation/prompts/coordinator_plan.md`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/prompts/coordinator_plan.md)

Output schema:

- `Problem Restatement`
- `Success Criteria`
- `Plan`
- `Risks and Unknowns`
- `Research Brief`

Local context view:

- `task_prompt`
- `task_domain`
- `benchmark_name`
- `benchmark_split`
- `task_type`
- `task_context`
- `task_objectives`
- `task_constraints`
- `answer_choices`
- `deliverable`
- `evaluation_criteria`
- `provided_references`

Role intent:

- translate the task into an operational brief
- define decomposition and criteria
- do not fully solve the task

### 6.2 Researcher

Prompt file:

- [`arch-1/implementation/prompts/researcher.md`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/prompts/researcher.md)

Output schema:

- `Evidence and Observations`
- `Options`
- `Assumptions`
- `Open Questions`
- `Research Handoff`

Local context view:

- all task-facing fields above
- `coordinator_plan`

Role intent:

- gather evidence and distinctions
- make assumptions explicit
- do not write the final answer

### 6.3 Analyst

Prompt file:

- [`arch-1/implementation/prompts/analyst.md`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/prompts/analyst.md)

Output schema:

- `Core Answer`
- `Reasoning`
- `Caveats`
- `Draft Deliverable`

Local context view:

- all task-facing fields above
- `coordinator_plan`
- `researcher_notes`

Role intent:

- synthesize the best available answer
- preserve important caveats
- produce a usable draft for critique

### 6.4 Critic

Prompt file:

- [`arch-1/implementation/prompts/critic.md`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/prompts/critic.md)

Output schema:

- `Major Issues`
- `Minor Issues`
- `Missing Evidence`
- `Required Revisions`

Local context view:

- all task-facing fields above
- `coordinator_plan`
- `researcher_notes`
- `analyst_draft`

Role intent:

- stress-test the draft
- identify overreach and missing support
- request concrete revisions

### 6.5 Coordinator Finalize

Prompt file:

- [`arch-1/implementation/prompts/coordinator_finalize.md`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/prompts/coordinator_finalize.md)

Output schema:

- `Final Answer`
- `Decision Rationale`
- `Residual Risks`

Local context view:

- all task-facing fields above
- `coordinator_plan`
- `researcher_notes`
- `analyst_draft`
- `critic_feedback`

Role intent:

- accept or reject parts of the draft
- integrate valid critique
- output one final answer

## 7. How The Code Enforces The Architecture

### 7.1 Config Loader

[`arch-1/implementation/committee_llm/config.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/config.py) is the contract layer.

It defines:

- `ROLE_SEQUENCE`
- `BASE_ROLE_ORDER`
- default role schemas
- default local context views
- `ExperimentConfig`
- `paper_claim_blockers()`

It enforces:

- the exact stage sequence
- presence of required roles
- non-empty schema declarations
- non-empty local context views
- role specialization metadata
- checkpoint uniqueness checks when the config claims post-trained specialization

### 7.2 Orchestrator

[`arch-1/implementation/committee_llm/orchestrator.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/orchestrator.py) is the runtime pipeline executor.

It is responsible for:

- constructing task blocks from benchmark metadata
- building role-local context views
- appending explicit instructions for each stage
- forcing exact schema headers in order
- making one model call per stage
- validating each stage response against the declared schema
- extracting the final answer from `Final Answer`
- computing trace summaries and behavior metrics

### 7.3 Schema Validation

The orchestrator parses stage outputs using known headers and marks each stage as:

- valid if all required headers are present with content
- invalid if any required section is missing

Per-stage validation is written into the trace output.

### 7.4 Trace Logging

Each committee trace records:

- role name
- role specialization metadata
- checkpoint metadata
- model name
- temperature
- max token setting
- system prompt
- local context view
- output schema
- schema validation result
- user prompt
- raw response
- elapsed seconds
- token usage

### 7.5 Behavior Metrics

The committee traces currently compute:

- `decomposition_rate`
- `evidence_grounding_rate`
- `revision_after_critique_rate`
- `premature_finalization_rate`
- `unsupported_claim_rate`

These are heuristic metrics derived from stage outputs, not human labels.

## 8. Baselines

The matched baselines are implemented in [`arch-1/implementation/committee_llm/baselines.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/baselines.py).

Baseline systems:

- `single_shot`
- `single_multiturn`
- `single_structured`

Definitions:

- `single_shot`: one direct response with a brief justification
- `single_multiturn`: three turns, plan then evidence then answer
- `single_structured`: one response with internal headings for decomposition, evidence, tentative answer, self-critique, and final answer

These baselines are critical because they test whether the committee structure adds value over simpler uses of the same underlying model family.

## 9. Evaluation Harness

The main runner is [`arch-1/implementation/committee_llm/benchmark.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/benchmark.py).

It supports:

- loading JSONL tasks
- running one or all systems
- per-task output traces
- resume mode
- continue-on-error mode
- benchmark-level summaries
- comparison metrics between committee and baselines

The systems are ordered as:

- `single_shot`
- `single_multiturn`
- `single_structured`
- `committee`

Benchmark outputs include:

- per-system completed and failed counts
- wall-clock elapsed time
- mean task time
- token totals
- model call counts
- reference metrics
- behavior metrics
- committee-vs-baseline comparison ratios

## 10. Config Inventory

The checked-in config set spans local, Modal, Colab-style local OpenAI-compatible, and template environments.

### 10.1 Local Ollama

- [`qwen3_8b_ollama_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_ollama_eval.json)

Key settings:

- endpoint: `http://127.0.0.1:11434/v1`
- model: `qwen3:8b`
- timeout: `600s`
- scaffold-only

### 10.2 Local / Self-hosted OpenAI-compatible

- [`qwen3_8b_committee.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_committee.json)
- [`qwen3_8b_committee_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_committee_eval.json)
- [`qwen3_30b_committee.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_30b_committee.json)
- [`qwen3_30b_committee_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_30b_committee_eval.json)

Key settings:

- endpoint: `http://localhost:8000/v1`
- scaffold-only
- Qwen3 8B and Qwen3 30B variants

### 10.3 Modal

- [`qwen3_8b_modal_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_modal_eval.json)
- [`qwen3_8b_modal_t4_awq_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_modal_t4_awq_eval.json)

Key settings:

- `Qwen/Qwen3-8B` on Modal
- `Qwen/Qwen3-8B-AWQ` on Modal T4
- scaffold-only

### 10.4 Colab-style / Local T4 Endpoint

- [`qwen3_8b_colab_t4_eval.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_8b_colab_t4_eval.json)

Key settings:

- endpoint: `http://127.0.0.1:8000/v1`
- model: `Qwen/Qwen3-8B-AWQ`
- timeout: `1800s`
- scaffold-only

### 10.5 Paper Template

- [`qwen3_arch1_posttrained_template.json`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs/qwen3_arch1_posttrained_template.json)

Key settings:

- `specialization_source = post_trained_role_specialists`
- role-specific checkpoint names
- intended for paper-valid runs once real checkpoints exist

## 11. Per-Role Runtime Defaults

Common recurring settings in the checked-in Qwen3 8B scaffold eval configs:

- coordinator plan: `max_tokens = 1200`
- researcher: `max_tokens = 1600`
- analyst: `max_tokens = 1400`
- critic: `max_tokens = 1200`
- coordinator finalize: `max_tokens = 1400`
- default evaluation temperatures: `0.0`

Thinking-enabled `extra_body` is used in several configs on:

- `coordinator_plan`
- `coordinator_finalize`

through:

- `chat_template_kwargs.enable_thinking = true`

## 12. Deployment Environments

### 12.1 Modal L4 Deployment

Deployment entrypoint:

- [`arch-1/implementation/modal_vllm_qwen3.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/modal_vllm_qwen3.py)

Important settings:

- app name: `korg-arch1-qwen3-vllm`
- model: `Qwen/Qwen3-8B`
- GPU: `L4`
- server: `vllm.entrypoints.openai.api_server`
- max model length: `8192`
- GPU memory utilization: `0.90`
- vLLM version: `0.10.2`
- transformers version: `4.56.1`

### 12.2 Modal T4 AWQ Deployment

Deployment entrypoint:

- [`arch-1/implementation/modal_vllm_qwen3_t4_awq.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/modal_vllm_qwen3_t4_awq.py)

Important settings:

- app name: `korg-arch1-qwen3-t4-awq-vllm`
- model: `Qwen/Qwen3-8B-AWQ`
- GPU: `T4`
- quantization: `AWQ`
- max model length: `8192`
- max sequences: `16`
- GPU memory utilization: `0.85`

### 12.3 AWS g4dn Setup

Environment notes are in:

- [`arch-1/aws/SETUP.md`](/Users/luthiraa/Documents/korg2.4/arch-1/aws/SETUP.md)
- [`arch-1/aws/run_gpqa_aws_use1.sh`](/Users/luthiraa/Documents/korg2.4/arch-1/aws/run_gpqa_aws_use1.sh)

Recorded AWS setup state on `2026-04-05`:

- region: `us-east-2`
- instance type target: `g4dn.xlarge`
- market: `spot`
- AMI: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403`
- root volume: `150 GB gp3`
- alternate prepared region: `us-east-1`

The benchmark script:

- unpacks the archived `arch-1` workspace
- installs vLLM and dependencies
- launches a local OpenAI-compatible endpoint
- waits for `/v1/models`
- runs the benchmark harness
- optionally syncs outputs to S3
- shuts the instance down afterward

## 13. Benchmark Protocol

Canonical benchmark planning docs:

- [`arch-1/docs/BENCHMARKS.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/BENCHMARKS.md)
- [`arch-1/docs/REMOTE_EVAL_PLAN.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/REMOTE_EVAL_PLAN.md)

Planned benchmark structure:

- main benchmark: `HotpotQA`, target `100` examples
- secondary benchmark: `FRAMES`, only if real retrieval is enabled
- stress benchmark: custom set, target `50` examples

Required systems:

- `single_shot`
- `single_multiturn`
- `single_structured`
- `committee`

Planned metrics:

- answer accuracy
- answer F1
- supporting-fact title precision
- supporting-fact title recall
- supporting-fact context coverage
- prompt tokens
- completion tokens
- total tokens
- model calls
- wall-clock latency
- decomposition rate
- evidence-grounding rate
- revision-after-critique rate
- premature-finalization rate
- unsupported-claim rate

Required ablations in the planning docs:

- full committee
- no critic
- no researcher
- no analyst
- order swap: coordinator -> analyst -> researcher -> critic -> coordinator
- order swap: coordinator -> researcher -> critic -> analyst -> coordinator
- budget-matched single-model baseline

Important current gap:

- the planning docs specify ablations, but the checked-in run set does not yet contain the full ablation matrix

## 14. Recorded Benchmark Timeline

### 14.1 2026-04-03: Early HotpotQA and GPQA pilots

Key runs:

- `hotpotqa_modal_committee_5pilot_20260403`
- `gpqa_diamond_modal_committee_5pilot_20260403`

Purpose:

- validate end-to-end committee execution on small slices before larger spend

### 14.2 2026-04-03: HotpotQA smoke comparison

Key run:

- `hotpotqa_modal_smoke_2`

Purpose:

- compare all four systems on a tiny two-task slice

### 14.3 2026-04-05: Full GPQA Modal committee attempt

Key run:

- `gpqa_diamond_modal_committee_198_20260405`

Purpose:

- first large committee run on the full GPQA Diamond set

Outcome:

- only partial completion because of billing-cap and infrastructure failures

### 14.4 2026-04-05: Modal T4 AWQ GPQA attempt

Key run:

- `gpqa_modal_t4_awq_thinking_198_20260405`

Purpose:

- evaluate the AWQ T4 path

Outcome:

- only `3/198` successful traces

### 14.5 2026-04-07: Strongest HotpotQA committee run

Key run:

- `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b`

Purpose:

- medium-scale committee benchmark on HotpotQA using the AWQ T4-class endpoint path

Outcome:

- strongest checked-in Architecture 1 quality result in the repo

### 14.6 2026-04-14: Matched single-shot HotpotQA baseline

Key run:

- `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414`

Purpose:

- direct baseline comparison against the 75-task HotpotQA committee run

Outcome:

- much cheaper than committee
- far worse on answer quality

## 15. Canonical Results

### 15.1 HotpotQA Smoke Run

Artifact:

- [`arch-1/docs/BENCHMARK_RESULTS.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/BENCHMARK_RESULTS.md)

Run:

- `hotpotqa_modal_smoke_2`
- date generated: `2026-04-03T17:55:21+00:00`
- task count: `2`

Results:

| System | Completed | Failed | Mean sec/task | Total tokens | Answer EM | Answer F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| committee | 2 | 0 | 232.20 | 39,229 | 0.50 | 0.6364 |
| single_multiturn | 2 | 0 | 87.45 | 12,847 | 0.00 | 0.0156 |
| single_shot | 2 | 0 | 167.54 | 4,642 | 0.00 | 0.0090 |
| single_structured | 2 | 0 | 45.11 | 4,581 | 0.00 | 0.0082 |

Committee behavior metrics on this smoke run:

- decomposition rate: `1.0`
- evidence grounding rate: `1.0`
- premature finalization rate: `0.0`
- revision after critique rate: `1.0`
- unsupported claim rate: `0.0`

### 15.2 HotpotQA 5-task Committee Pilot

Run:

- `hotpotqa_modal_committee_5pilot_20260403`

Results:

- completed: `5/5`
- answer EM: `0.20`
- answer F1: `0.4375`
- mean sec/task: `244.35`
- total tokens: `101,189`

Behavior metrics:

- decomposition rate: `0.8`
- evidence grounding rate: `1.0`
- premature finalization rate: `0.0`
- revision after critique rate: `1.0`
- unsupported claim rate: `0.0`

### 15.3 Strongest HotpotQA Committee Benchmark

Run:

- `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b`

Summary source:

- [`arch-1/evals/runs/hotpotqa_aws_g4dn_thinking_75_20260407_retry4b/summary.json`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/hotpotqa_aws_g4dn_thinking_75_20260407_retry4b/summary.json)

Results:

- completed: `73/75`
- failed: `2/75`
- answer EM: `0.3835616438`
- answer F1: `0.5774265533`
- supporting-fact context coverage: `1.0`
- supporting-fact title precision: `0.6038812785`
- supporting-fact title recall: `0.6038812785`
- mean sec/task: `150.04`
- wall-clock elapsed: `11,266.94s`
- total model calls: `365`
- total tokens: `1,543,424`
- prompt tokens: `1,270,025`
- completion tokens: `273,399`

Behavior metrics:

- decomposition rate: `0.8767123288`
- evidence grounding rate: `0.9863013699`
- premature finalization rate: `0.0`
- revision after critique rate: `1.0`
- unsupported claim rate: `0.0`

Interpretation:

- this is the clearest evidence in the repo that the scaffolded committee helps on HotpotQA
- it does so at a large compute cost

### 15.4 Matched 75-task HotpotQA Single-shot Baseline

Run:

- `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414`

Summary source:

- [`arch-1/evals/runs/hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/summary.json`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/summary.json)

Results:

- completed: `75/75`
- answer EM: `0.0`
- answer F1: `0.0161567665`
- supporting-fact context coverage: `1.0`
- supporting-fact title precision: `0.8977777778`
- supporting-fact title recall: `0.8977777778`
- mean sec/task: `23.06`
- wall-clock elapsed: `1,730.23s`
- total model calls: `75`
- total tokens: `174,866`
- prompt tokens: `132,947`
- completion tokens: `41,919`

Direct committee-vs-single-shot comparison on the same 75-task Hotpot slice from the checked-in report:

- committee won answer F1 on `68` tasks
- single-shot won on `4`
- `1` task tied
- committee used about `8.8x` the total tokens
- committee used about `6.5x` the mean per-task latency

### 15.5 HotpotQA Modal Single-multiturn Partial Run

Run:

- `hotpotqa_modal_100`

Observed checked-in state:

- only the `single_multiturn` system summary is present
- task count in the summary is `26`

Results:

- completed: `26/26`
- answer EM: `0.0`
- answer F1: `0.0158443860`
- mean sec/task: `104.72`
- total tokens: `192,880`

Interpretation:

- this is not a full four-system 100-task benchmark artifact
- it should be treated as a partial run rather than a complete benchmark table

### 15.6 GPQA 5-task Committee Pilot

Run:

- `gpqa_diamond_modal_committee_5pilot_20260403`

Results from the generated benchmark report:

- completed: `5/5`
- choice accuracy: `80.0%`
- mean sec/task: `366.7`
- total tokens: `95,942`

Interpretation:

- promising early signal
- too small to support strong conclusions

### 15.7 Full GPQA Modal Committee Attempt

Main report:

- [`arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md)

Run root:

- [`arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405)

Aggregate results:

- completed: `163/198`
- failed: `35/198`
- successful-trace accuracy: `84 / 163 = 51.53%`
- conservative lower-bound full-set accuracy: `84 / 198 = 42.42%`
- mean sec/task on successful traces: `383.16`
- mean tokens/task on successful traces: `20,619.74`
- total model calls: `815`
- total tokens: `3,361,018`
- prompt tokens: `2,412,647`
- completion tokens: `948,371`
- wall-clock span: `2026-04-05T08:08:10Z` to `2026-04-05T12:34:40Z`

Shard breakdown:

| Shard | Success | Failed | Accuracy | Mean sec/task | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_shard00 | 42 | 8 | 45.24% | 373.11 | 851,370 |
| fixed_shard01 | 41 | 9 | 60.98% | 383.25 | 847,827 |
| fixed_shard02 | 38 | 12 | 47.37% | 402.44 | 813,336 |
| fixed_shard03 | 42 | 6 | 52.38% | 375.68 | 848,485 |

Failure breakdown:

- `30` tasks: `HTTP 429` billing-cap rejection from Modal
- `4` tasks: `HTTP 500` Modal internal termination
- `1` task: `HTTP 400` context-window overflow from requested `max_tokens`

Context against reference numbers recorded in the report:

- official `Qwen3-8B` non-thinking GPQA Diamond: `39.3`
- official `Qwen3-8B` thinking GPQA Diamond: `62.0`

Interpretation:

- the completed traces beat the recorded non-thinking reference
- the run remains below the recorded thinking reference
- because the run was incomplete, the conservative whole-benchmark interpretation is much weaker

### 15.8 GPQA Modal T4 AWQ Attempt

Run:

- `gpqa_modal_t4_awq_thinking_198_20260405`

Results:

- completed: `3/198`
- failed: `195`
- choice accuracy on successful traces: `33.33%`
- mean sec/task: `1365.47`
- total tokens: `53,509`

Interpretation:

- this path was not operationally stable enough to serve as the main benchmark route

## 16. Full Run Catalog

For the full checked-in run inventory, use:

- [`arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md)

That report catalogs every top-level summary under `arch-1/evals/runs/*/summary.json` plus the aggregated four-shard GPQA run.

Notable catalog categories include:

- smoke runs
- pilot runs
- full or partial benchmark runs
- Modal probe runs
- retry attempts
- failed setup artifacts

## 17. Operational Lessons From The Recorded Runs

Major observed operational constraints:

- five-stage committee traces multiply failure surface area compared with one-call baselines
- accumulated context across stages makes later calls more expensive
- Modal billing limits were a real blocker on GPQA
- context budget tuning matters; at least one GPQA failure came from overflow
- some AWQ/Modal paths were fragile or misconfigured

Key practical takeaway:

- the architecture can materially improve answer quality on decomposition-heavy tasks
- the cost and infrastructure burden are large enough that deployment discipline matters as much as prompt design

## 18. Known Gaps And Caveats

### 18.1 No Checked-in Paper-valid Checkpoints

The repo does not currently ship the four distinct role-specialist checkpoints required for the full paper claim.

### 18.2 Missing Full Ablation Matrix

The benchmark planning docs require ablations, but the checked-in artifacts do not yet cover:

- no critic
- no researcher
- no analyst
- order swaps
- full budget-matched ablation grid

### 18.3 Mixed Benchmark Conditions

The best committee and best baseline artifacts are not all from one single unified all-systems summary file. The strongest comparisons are assembled from multiple checked-in runs and reports.

### 18.4 Some Artifacts Are Bring-up Artifacts

Some summary files are:

- probes
- smokes
- retries
- setup failures

They are still useful operationally, but they should not all be treated as headline benchmark evidence.

## 19. Reproduction Workflow

### 19.1 Local Test Workflow

Recommended commands from the docs:

```bash
cd /Users/luthiraa/Documents/korg2.4/arch-1
export PYTHONPATH=implementation
export OPENAI_API_KEY=EMPTY
python -m unittest discover -s evals/tests
python -m committee_llm.benchmark_data --suite hotpotqa --input evals/data/hotpot_dev_distractor_v1.json --output evals/data/benchmarks/hotpotqa_dev.jsonl --limit 100
python -m committee_llm.benchmark --config implementation/configs/qwen3_8b_ollama_eval.json --tasks evals/data/benchmarks/hotpotqa_dev.jsonl --outdir evals/runs/hotpotqa_arch1_local --system all --limit 100
python -m committee_llm.benchmark_report --summary evals/runs/hotpotqa_arch1_local/summary.json --output docs/BENCHMARK_RESULTS.md
```

### 19.2 Modal Deployment Workflow

Deploy either:

```bash
modal deploy arch-1/implementation/modal_vllm_qwen3.py
```

or:

```bash
modal deploy arch-1/implementation/modal_vllm_qwen3_t4_awq.py
```

Then run benchmark commands against the matching config.

### 19.3 AWS Workflow

Use:

- [`arch-1/aws/run_gpqa_aws_use1.sh`](/Users/luthiraa/Documents/korg2.4/arch-1/aws/run_gpqa_aws_use1.sh)

The script assumes:

- archived workspace transfer to the instance
- local vLLM serving on port `8000`
- benchmark output under `evals/runs/<run_id>`
- optional S3 sync

## 20. Verification Notes For This Documentation Pass

This document was prepared from:

- the checked-in architecture docs
- benchmark planning docs
- implementation code
- config files
- Modal deployment scripts
- AWS setup scripts
- benchmark summary JSON files
- generated benchmark reports
- GPQA-specific benchmark report notes

Additional verification performed during this pass:

- the `arch-1/evals/tests` suite was invoked with the host `python3`

Observed result:

- test discovery failed under the host Python because it is Python `3.9`, and the codebase uses `dataclass(slots=True)`, which requires a newer interpreter than that runtime path provides

Practical interpretation:

- this is an environment mismatch on the machine-level `python3` used for the check
- it does not by itself invalidate the recorded benchmark artifacts

## 21. Recommended Next Steps

If the goal is a publishable Architecture 1 control section, the next concrete steps are:

1. train or provision the real role-specialist checkpoints and replace scaffold configs with paper-valid configs
2. run the missing ablation matrix on a stable endpoint budget
3. produce a unified same-slice comparison where all four systems run on the same task set and environment
4. rerun the failed GPQA tasks on stable infrastructure and merge only missing traces
5. keep the engineering scaffold report separate from the paper-valid result table

## 22. Primary Source Index

Architecture and docs:

- [`arch-1/docs/README.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/README.md)
- [`arch-1/docs/ARCHITECTURE_1.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/ARCHITECTURE_1.md)
- [`arch-1/docs/BENCHMARKS.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/BENCHMARKS.md)
- [`arch-1/docs/REMOTE_EVAL_PLAN.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/REMOTE_EVAL_PLAN.md)
- [`arch-1/docs/IMPLEMENTATION_NOTE.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/IMPLEMENTATION_NOTE.md)

Implementation:

- [`arch-1/implementation/committee_llm/config.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/config.py)
- [`arch-1/implementation/committee_llm/orchestrator.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/orchestrator.py)
- [`arch-1/implementation/committee_llm/baselines.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/baselines.py)
- [`arch-1/implementation/committee_llm/benchmark.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/committee_llm/benchmark.py)

Configs:

- [`arch-1/implementation/configs`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/configs)

Deployment:

- [`arch-1/implementation/modal_vllm_qwen3.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/modal_vllm_qwen3.py)
- [`arch-1/implementation/modal_vllm_qwen3_t4_awq.py`](/Users/luthiraa/Documents/korg2.4/arch-1/implementation/modal_vllm_qwen3_t4_awq.py)
- [`arch-1/aws/SETUP.md`](/Users/luthiraa/Documents/korg2.4/arch-1/aws/SETUP.md)
- [`arch-1/aws/run_gpqa_aws_use1.sh`](/Users/luthiraa/Documents/korg2.4/arch-1/aws/run_gpqa_aws_use1.sh)

Benchmark reports:

- [`arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md`](/Users/luthiraa/Documents/korg2.4/arch-1/docs/generated/arch1_benchmark_report/ARCHITECTURE1_BENCHMARK_REPORT.md)
- [`arch-1/evals/reports/benchmark_comparison_summary.md`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/reports/benchmark_comparison_summary.md)
- [`arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md`](/Users/luthiraa/Documents/korg2.4/arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md)
