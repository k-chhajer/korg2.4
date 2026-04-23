# Benchmark Results

- Updated at (UTC): `2026-04-15T16:20:00+00:00`
- Benchmark: HotpotQA Distractor, first `75` examples from `arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl`
- Hardware: AWS `g4dn.xlarge`, Qwen3-8B-AWQ via local vLLM
- Primary metric: `answer_f1`

## Source Summaries

- `arch-1/evals/runs/hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/summary.json`
- `s3://korg24-arch1-gpqa-20260406-525184038455/hotpotqa_aws_g4dn_qwen_single_shot_thinking_75_20260415/summary.json`
- `arch-1/evals/runs/hotpotqa_aws_g4dn_thinking_75_20260407_retry4b/summary.json`

## Run Overview

| Run | System | Thinking | Completed | Failed | Mean Run (s) | Tasks/s | Total Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414` | single_shot | no | 75 | 0 | 23.0636 | 0.0433 | 174866 |
| `hotpotqa_aws_g4dn_qwen_single_shot_thinking_75_20260415` | single_shot | yes | 75 | 0 | 21.5607 | 0.0464 | 174866 |
| `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b` | committee | partial role-level thinking | 73 | 2 | 150.0377 | 0.0065 | 1543424 |

## HotpotQA Metrics

| Run | answer_em | answer_f1 | supporting_fact_context_coverage | supporting_fact_title_precision | supporting_fact_title_recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414` | 0.0000 | 0.0162 | 1.0000 | 0.8978 | 0.8978 |
| `hotpotqa_aws_g4dn_qwen_single_shot_thinking_75_20260415` | 0.0000 | 0.0162 | 1.0000 | 0.8978 | 0.8978 |
| `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b` | 0.3836 | 0.5774 | 1.0000 | 0.6039 | 0.6039 |

## Conservative Full-Set Scores

The committee run completed `73/75` examples. The table below treats the two failed examples as zero-score examples for answer metrics.

| Run | Completed-Traces answer_em | Full-Set answer_em | Completed-Traces answer_f1 | Full-Set answer_f1 |
| --- | ---: | ---: | ---: | ---: |
| `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414` | 0.0000 | 0.0000 | 0.0162 | 0.0162 |
| `hotpotqa_aws_g4dn_qwen_single_shot_thinking_75_20260415` | 0.0000 | 0.0000 | 0.0162 | 0.0162 |
| `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b` | 0.3836 | 0.3733 | 0.5774 | 0.5620 |

## Behavior Metrics

Behavior metrics are recorded for the committee pipeline only.

| Run | decomposition_rate | evidence_grounding_rate | premature_finalization_rate | revision_after_critique_rate | unsupported_claim_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa_aws_g4dn_thinking_75_20260407_retry4b` | 0.8767 | 0.9863 | 0.0000 | 1.0000 | 0.0000 |

## Notes

- The April 15 single-shot thinking trace includes `chat_template_kwargs.enable_thinking=true`; sampled outputs begin with `<think>`, confirming the Qwen3 thinking path was active.
- Single-shot thinking and single-shot non-thinking produced the same aggregate answer metrics and token totals on this benchmark slice.
- The committee run remains much slower and more token-intensive, but it is the only run in this set with materially nonzero HotpotQA answer accuracy.
- The committee result should be reported with both completed-trace metrics and conservative full-set metrics because `2/75` examples failed.
