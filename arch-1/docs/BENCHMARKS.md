# Benchmark Protocol

This evaluation plan is intentionally small and thesis-linked. The goal is to test decomposition, evidence use, critique, revision, and synthesis without bloating runtime.

Main benchmark:

- HotpotQA, 100 examples

Secondary benchmark:

- FRAMES only if the researcher role performs actual retrieval beyond provided context
- current local scaffold does not do live retrieval, so FRAMES is not enabled yet

Stress benchmark:

- custom set, 50 examples
- should target unsupported-jump failures, premature finalization, weak evidence use, and critique-sensitive revisions

Systems to compare:

- `single_shot`
- `single_multiturn`
- `single_structured`
- `committee`

Metrics to report:

- answer accuracy
- answer F1 where applicable
- supporting-fact title recall / precision / context coverage on HotpotQA
- total input tokens
- total output tokens
- total tokens
- number of model calls
- wall-clock latency
- decomposition rate
- evidence-grounding rate
- revision-after-critique rate
- premature-finalization rate
- unsupported-claim rate

Required ablations:

- full committee
- no critic
- no researcher
- no analyst
- order swap: coordinator -> analyst -> researcher -> critic -> coordinator
- order swap: coordinator -> researcher -> critic -> analyst -> coordinator
- budget-matched single-model baseline

Benchmarks should be run only after config verification:

- if `paper_claim_blockers` is non-empty, the run is a scaffold run, not a paper-valid post-trained committee result

Recommended local commands:

```powershell
cd C:\Users\luthi\Documents\korg\korg2_4_remote\arch-1
$env:PYTHONPATH = "implementation"
$env:OPENAI_API_KEY = "EMPTY"
python -m unittest discover -s evals\tests
python -m committee_llm.benchmark_data --suite hotpotqa --input evals\data\hotpot_dev_distractor_v1.json --output evals\data\benchmarks\hotpotqa_dev.jsonl --limit 100
python -m committee_llm.benchmark --config implementation\configs\qwen3_8b_ollama_eval.json --tasks evals\data\benchmarks\hotpotqa_dev.jsonl --outdir evals\runs\hotpotqa_arch1_local --system all --limit 100
python -m committee_llm.benchmark_report --summary evals\runs\hotpotqa_arch1_local\summary.json --output docs\BENCHMARK_RESULTS.md
```

Compute-friendly recommendation:

- run HotpotQA first
- add the custom stress set second
- do ablations on a smaller subset after the main trend is established
- do not run FRAMES until retrieval is truly part of the pipeline
