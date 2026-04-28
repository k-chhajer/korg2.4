# Benchmark Results

- Summary source: `C:\Users\luthi\Documents\korg\korg2_4_remote\arch-1\evals\runs\hotpotqa_modal_smoke_2\summary.json`
- Generated at (UTC): `2026-04-03T17:55:21+00:00`
- Config: `qwen3_8b_modal_eval`
- Config path: `C:\Users\luthi\Documents\korg\korg2_4_remote\arch-1\implementation\configs\qwen3_8b_modal_eval.json`
- Task count: `2`

## Verification

Paper-claim blockers:

- Role specialization source is not set to post_trained_role_specialists, so this config should be treated as a prompt scaffold.

## System Overview

| System | Completed | Failed | Wall Clock (s) | Mean Run (s) | Tasks/s | Total Tokens | Model Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| committee | 2 | 0 | 464.4035 | 232.2001 | 0.0043 | 39229 | 10 |
| single_multiturn | 2 | 0 | 174.9127 | 87.451 | 0.0114 | 12847 | 6 |
| single_shot | 2 | 0 | 335.1067 | 167.5434 | 0.006 | 4642 | 2 |
| single_structured | 2 | 0 | 90.2417 | 45.1146 | 0.0222 | 4581 | 2 |

## Overall Metrics

| System | answer_em | answer_f1 | supporting_fact_context_coverage | supporting_fact_title_precision | supporting_fact_title_recall |
| --- | --- | --- | --- | --- | --- |
| committee | 0.5 | 0.6364 | 1 | 0.5 | 0.5 |
| single_multiturn | 0 | 0.0156 | 1 | 0.8333 | 0.8333 |
| single_shot | 0 | 0.009 | 1 | 1 | 1 |
| single_structured | 0 | 0.0082 | 1 | 1 | 1 |

## Per-Benchmark Metrics

| System | Benchmark | Count | answer_em | answer_f1 | supporting_fact_context_coverage | supporting_fact_title_precision | supporting_fact_title_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| committee | hotpotqa_distractor | 2 | 0.5 | 0.6364 | 1 | 0.5 | 0.5 |
| single_multiturn | hotpotqa_distractor | 2 | 0 | 0.0156 | 1 | 0.8333 | 0.8333 |
| single_shot | hotpotqa_distractor | 2 | 0 | 0.009 | 1 | 1 | 1 |
| single_structured | hotpotqa_distractor | 2 | 0 | 0.0082 | 1 | 1 | 1 |

## Behavior Metrics

| System | decomposition_rate | evidence_grounding_rate | premature_finalization_rate | revision_after_critique_rate | unsupported_claim_rate |
| --- | --- | --- | --- | --- | --- |
| committee | 1 | 1 | 0 | 1 | 0 |
| single_multiturn | N/A | N/A | N/A | N/A | N/A |
| single_shot | N/A | N/A | N/A | N/A | N/A |
| single_structured | N/A | N/A | N/A | N/A | N/A |

## Comparison

| Comparison | Latency Ratio | Token Ratio | answer_em Delta | answer_f1 Delta | supporting_fact_context_coverage Delta | supporting_fact_title_precision Delta | supporting_fact_title_recall Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| committee_vs_single_multiturn | 2.6552 | 3.0536 | 0.5 | 0.6207 | 0 | -0.3333 | -0.3333 |
| committee_vs_single_shot | 1.3859 | 8.4509 | 0.5 | 0.6273 | 0 | -0.5 | -0.5 |
| committee_vs_single_structured | 5.1469 | 8.5634 | 0.5 | 0.6281 | 0 | -0.5 | -0.5 |

## Failures

_No failures were recorded in this summary._
