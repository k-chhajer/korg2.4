# Benchmark Results

- Summary source: `C:\Dev\grok_multiagent\arch-1\.test-benchmark-report-616247d5774444aba1dbfe05702c19c5\summary.json`
- Generated at (UTC): `2026-04-03T23:45:04+00:00`
- Config: `qwen3_8b_committee_eval`
- Config path: `C:\Dev\grok_multiagent\arch-1\implementation\configs\qwen3_8b_committee_eval.json`
- Task count: `2`

## Verification

Paper-claim blockers:

- Role specialization source is not set to post_trained_role_specialists, so this config should be treated as a prompt scaffold.

## System Overview

| System | Completed | Failed | Wall Clock (s) | Mean Run (s) | Tasks/s | Total Tokens | Model Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| committee | 2 | 1 | 5 | 2.5 | 0.4 | 200 | 10 |
| single_shot | 2 | 0 | 3 | 1.5 | 0.6667 | 80 | 2 |

## Overall Metrics

| System | answer_em | answer_f1 |
| --- | --- | --- |
| committee | 0.5 | 0.75 |
| single_shot | 0 | 0.25 |

## Per-Benchmark Metrics

| System | Benchmark | Count | answer_em | answer_f1 |
| --- | --- | --- | --- | --- |
| committee | hotpotqa_distractor | 2 | 0.5 | 0.75 |
| single_shot | hotpotqa_distractor | 2 | 0 | 0.25 |

## Behavior Metrics

| System | decomposition_rate |
| --- | --- |
| committee | 1 |
| single_shot | N/A |

## Comparison

| Comparison | Latency Ratio | Token Ratio | answer_em Delta | answer_f1 Delta |
| --- | --- | --- | --- | --- |
| committee_vs_single_shot | 1.6667 | 2.5 | 0.5 | 0.5 |

## Failures

| System | Task ID | Error |
| --- | --- | --- |
| committee | task-3 | backend timeout |
