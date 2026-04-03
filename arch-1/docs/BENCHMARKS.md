# Benchmark Protocol

This harness is now set up to benchmark both:

- `committee_v1_explicit_coordination`
- `single_direct_baseline`

The intended publication-facing use is to compare quality and latency side by side on canonical benchmarks, while keeping the underlying model fixed.

## Recommended Benchmarks

- HotpotQA distractor dev: multi-hop reading comprehension with official answer EM/F1. Official site: https://hotpotqa.github.io/
- MuSiQue-Ans dev: compositional multi-hop QA with normalized answer EM/F1. Official repo: https://github.com/StonyBrookNLP/musique
- GPQA: hard graduate-level multiple-choice science QA with accuracy. Paper page: https://arxiv.org/abs/2311.12022

## Research Setup

For a publishable benchmark table:

1. Use the same base model across all systems.
2. Use deterministic decoding or as close as your backend allows.
3. Report quality metrics and latency together.
4. Report total token usage from the backend trace.
5. Keep benchmark context intact for reading-comprehension tasks.
6. Do not mix closed-book and provided-context variants in one result table.

## Suggested Config

Use the evaluation configs in [qwen3_8b_committee_eval.json](/C:/Users/luthi/Documents/korg/arch-1/implementation/configs/qwen3_8b_committee_eval.json) or [qwen3_30b_committee_eval.json](/C:/Users/luthi/Documents/korg/arch-1/implementation/configs/qwen3_30b_committee_eval.json) so every role runs at `temperature = 0.0`.

## Convert Official Data

HotpotQA:

```powershell
cd C:\Users\luthi\Documents\korg\arch-1
$env:PYTHONPATH = "implementation"
python -m committee_llm.benchmark_data --suite hotpotqa --input evals\data\hotpot_dev_distractor_v1.json --output evals\data\benchmarks\hotpotqa_dev.jsonl
```

MuSiQue:

```powershell
python -m committee_llm.benchmark_data --suite musique --input evals\data\musique_ans_v1.0_dev.jsonl --output evals\data\benchmarks\musique_dev.jsonl
```

GPQA:

```powershell
python -m committee_llm.benchmark_data --suite gpqa --input evals\data\gpqa_diamond.csv --output evals\data\benchmarks\gpqa_diamond.jsonl
```

## Run Benchmark Comparison

```powershell
python -m committee_llm.benchmark --config implementation\configs\qwen3_8b_committee_eval.json --tasks evals\data\benchmarks\hotpotqa_dev.jsonl --outdir evals\runs\hotpotqa_qwen8b --system both
```

The benchmark summary includes:

- per-system mean latency
- per-system token usage
- per-system benchmark metrics
- committee-vs-single metric deltas
- committee-vs-single latency and token ratios

## Render Markdown Report

After a run finishes, render the JSON summary into a publication-facing Markdown file:

```powershell
python -m committee_llm.benchmark_report --summary evals\runs\hotpotqa_qwen8b\summary.json --output docs\BENCHMARK_RESULTS.md
```

The Markdown report includes:

- system-level throughput, latency, and token tables
- overall and per-benchmark metric tables
- comparison ratios and metric deltas
- explicit failure rows when a run did not complete cleanly

## Notes

- HotpotQA data can be downloaded from the official dataset link on the HotpotQA site.
- MuSiQue conversion expects the official JSON or JSONL files after you download them from the official repository instructions.
- GPQA distribution may require separately obtaining the official file; once you have it locally, the converter expects a CSV with one correct answer and three incorrect answers per row.
