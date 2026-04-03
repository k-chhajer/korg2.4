# Benchmark Results

- Summary source: `C:\Users\luthi\Documents\korg\arch-1\evals\runs\hotpotqa_qwen8b\summary.json`
- Generated at (UTC): `2026-03-31T16:49:15+00:00`
- Config: `qwen3_8b_committee_eval`
- Config path: `C:\Users\luthi\Documents\korg\arch-1\implementation\configs\qwen3_8b_committee_eval.json`
- Task count: `1`

## System Overview

| System | Completed | Failed | Wall Clock (s) | Mean Run (s) | Tasks/s | Total Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| committee | 0 | 1 | 4.1317 | N/A | 0 | N/A |
| single_direct | 0 | 1 | 4.0429 | N/A | 0 | N/A |

## Overall Metrics

_No completed benchmark metrics were recorded in this summary._

## Per-Benchmark Metrics

_No per-benchmark metrics were recorded in this summary._

## Comparison

| Comparison | Latency Ratio | Token Ratio |
| --- | --- | --- |
| committee_vs_single_direct | N/A | N/A |

## Failures

| System | Task ID | Error |
| --- | --- | --- |
| committee | 5a8b57f25542995d1e6f1371 | Could not reach chat backend at http://localhost:8000/v1/chat/completions: [WinError 10061] No connection could be made because the target machine actively refused it |
| single_direct | 5a8b57f25542995d1e6f1371 | Could not reach chat backend at http://localhost:8000/v1/chat/completions: [WinError 10061] No connection could be made because the target machine actively refused it |
