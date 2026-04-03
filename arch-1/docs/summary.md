# Benchmark Dataset Investigation Summary

## Request

The user asked which datasets were used for the benchmarks, then asked for a summary file describing everything that happened.

## What Was Checked

The repository was searched for references to:

- `benchmark`
- `benchmarks`
- `dataset`
- `datasets`
- `eval`
- `evaluation`

The main files inspected were:

- [BENCHMARKS.md](C:/Users/luthi/Documents/korg/arch-1/docs/BENCHMARKS.md)
- [README.md](C:/Users/luthi/Documents/korg/arch-1/docs/README.md)
- [benchmark_data.py](C:/Users/luthi/Documents/korg/arch-1/implementation/committee_llm/benchmark_data.py)
- [hotpotqa_dev.jsonl](C:/Users/luthi/Documents/korg/arch-1/evals/data/benchmarks/hotpotqa_dev.jsonl)
- [hotpotqa_dev_20.jsonl](C:/Users/luthi/Documents/korg/arch-1/evals/data/benchmarks/hotpotqa_dev_20.jsonl)
- [summary.json](C:/Users/luthi/Documents/korg/arch-1/evals/runs/hotpotqa_qwen3_8b_ollama_20_smoke/summary.json)

## Findings

The benchmark harness is set up to support three benchmark datasets:

1. HotpotQA
2. MuSiQue
3. GPQA

More specifically, the documented benchmark variants are:

- `hotpotqa_distractor` on the `dev` split
- `musique_ans` on the `dev` split
- `gpqa`

This is documented in `BENCHMARKS.md` and implemented in `committee_llm/benchmark_data.py`.

## What Was Actually Present And Used

In the checked-in workspace, the only official benchmark source file present was:

- `data/hotpot_dev_distractor_v1.json`

The converted benchmark task files present were:

- `data/benchmarks/hotpotqa_dev.jsonl`
- `data/benchmarks/hotpotqa_dev_20.jsonl`

The recorded benchmark run that was inspected used:

- benchmark name: `hotpotqa_distractor`
- split: `dev`
- source dataset: `HotpotQA`

The inspected run summary showed one completed task for both:

- `committee`
- `single_direct`

and reported HotpotQA answer metrics for each system.

## Conclusion

The codebase supports benchmarking on:

- HotpotQA
- MuSiQue
- GPQA

But the benchmark artifacts actually present and used in this repository were based on:

- HotpotQA distractor dev

## Notes

During repository search, `rg` reported access-denied errors for a couple of temporary paths, but those errors did not affect the benchmark-dataset conclusion.
