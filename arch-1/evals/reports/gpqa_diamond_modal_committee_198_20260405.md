# GPQA Diamond Committee Run on Modal

## Status

The full `198`-example `committee` run was started on April 5, 2026 and executed as four parallel shard runs against the Modal-hosted `Qwen/Qwen3-8B` endpoint.

The run did **not** complete all 198 examples. It was blocked by the Modal workspace billing-cycle cap:

- `163/198` examples completed successfully
- `35/198` examples failed before completion
- dominant blocker: `30` tasks failed with `HTTP 429: workspace billing cycle spend limit reached`

This report reflects the completed work only.

## Aggregate Results

- Successful-trace accuracy: `84 / 163 = 51.53%`
- Conservative lower-bound full-set accuracy: `84 / 198 = 42.42%`
- Successful traces written: `163`
- Failed tasks: `35`
- Mean elapsed time per successful task: `383.16s`
- Mean tokens per successful task: `20,619.74`
- Total model calls: `815`
- Total tokens: `3,361,018`
- Prompt tokens: `2,412,647`
- Completion tokens: `948,371`
- Parallel wall-clock span: `2026-04-05T08:08:10Z` to `2026-04-05T12:34:40Z`

Definitions:

- `Successful-trace accuracy` means accuracy measured only on the `163` examples that completed and produced a scoreable trace.
- `Conservative lower bound` means treating every unfinished example as wrong. Here that is `84` known-correct answers divided by the full `198`-example benchmark.

## Shard Breakdown

| Shard | Success | Failed | Choice Accuracy | Mean Run Sec | Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_shard00` | 42 | 8 | 45.24% | 373.11 | 851,370 |
| `fixed_shard01` | 41 | 9 | 60.98% | 383.25 | 847,827 |
| `fixed_shard02` | 38 | 12 | 47.37% | 402.44 | 813,336 |
| `fixed_shard03` | 42 | 6 | 52.38% | 375.68 | 848,485 |

## Failure Breakdown

- `30` tasks: `HTTP 429` billing-cap rejection from Modal
- `4` tasks: `HTTP 500` Modal internal termination
- `1` task: `HTTP 400` context-window overflow from requested `max_tokens`

Failed task IDs:

`gpqa-00043`, `gpqa-00044`, `gpqa-00045`, `gpqa-00046`, `gpqa-00047`, `gpqa-00048`, `gpqa-00049`, `gpqa-00050`, `gpqa-00092`, `gpqa-00093`, `gpqa-00094`, `gpqa-00095`, `gpqa-00096`, `gpqa-00097`, `gpqa-00098`, `gpqa-00099`, `gpqa-00100`, `gpqa-00128`, `gpqa-00136`, `gpqa-00141`, `gpqa-00142`, `gpqa-00143`, `gpqa-00144`, `gpqa-00145`, `gpqa-00146`, `gpqa-00147`, `gpqa-00148`, `gpqa-00149`, `gpqa-00150`, `gpqa-00193`, `gpqa-00194`, `gpqa-00195`, `gpqa-00196`, `gpqa-00197`, `gpqa-00198`

## Context Against Official GPQA Numbers

- `Qwen3-8B` non-thinking official GPQA Diamond: `39.3`
- `Qwen3-8B` thinking official GPQA Diamond: `62.0`
- `Grok 3 (Think)` official GPQA: `84.6`
- `Grok 4` official GPQA: `87.5`

Observed committee result from this blocked run:

- `51.53%` on the `163` completed traces
- `42.42%` conservative lower bound on the full `198` if all unfinished tasks are treated as wrong

## Artifacts

- Run root: `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405`
- Corrected config used for the real run: `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/qwen3_8b_modal_eval_timeout1800_abs.json`
- Per-shard summaries:
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/fixed_shard00/summary.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/fixed_shard01/summary.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/fixed_shard02/summary.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/fixed_shard03/summary.json`

## Recommended Next Step

After the Modal billing cap is lifted, rerun only the failed task IDs rather than restarting all 198 examples. For the retry pass:

- reduce per-stage `max_tokens` to avoid the context-overflow failure on `gpqa-00128`
- use lower concurrency or a fresh endpoint budget to avoid immediate `429` rejections
- keep the same successful traces and merge retry outputs into this run root

## Retry Attempt

I prepared a dedicated retry set containing the `35` failed task IDs and a reduced-budget retry config:

- Retry tasks: `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_failed_35.jsonl`
- Retry config: `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/qwen3_8b_modal_eval_retry400_abs.json`

A 1-task retry smoke test was attempted on `gpqa-00043` and failed immediately with the same endpoint blocker:

- `HTTP 429: modal-http: Webhook failed: workspace billing cycle spend limit reached`

Smoke-test artifact:

- `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_smoke/summary.json`

## Local Retry of Real Failures

I brought up a local OpenAI-compatible `llama-server` endpoint backed by the cached Qwen3 8B GGUF blob and reran the `5` non-budget failure IDs only:

- `gpqa-00043`
- `gpqa-00128`
- `gpqa-00136`
- `gpqa-00141`
- `gpqa-00193`

Artifacts:

- Local retry summary: `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/summary.json`
- Local retry outputs:
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/committee/gpqa-00043.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/committee/gpqa-00128.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/committee/gpqa-00136.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/committee/gpqa-00141.json`
  - `/Users/luthiraa/Documents/korg2.4/arch-1/evals/runs/gpqa_diamond_modal_committee_198_20260405/retry_real_failures_5_local_run/committee/gpqa-00193.json`

Observed result:

- Local rerun completed all `5/5` tasks without transport errors
- Measured local retry accuracy: `0 / 5 = 0.0%`
- Mean local retry elapsed time: `374.24s`
- Mean local retry tokens per task: `6,806.4`

Important caveat:

- The local traces are structurally abnormal for this benchmark harness. Most stages report nonzero token usage but have empty parsed `response` fields, and most task files end with an empty `final_answer`.
- Because of that parser/output mismatch, these local reruns should not be treated as clean recoveries of the Modal failures.

Practical effect on the benchmark:

- The original Modal result does **not** improve from this local rerun.
- If these five local tasks are naively merged in as completed-but-wrong, completed-trace accuracy becomes `84 / 168 = 50.00%`.
- The full-set lower bound remains `84 / 198 = 42.42%`.
