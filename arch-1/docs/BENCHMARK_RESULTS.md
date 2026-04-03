# Benchmark Results

No paper-valid Architecture 1 benchmark results are checked in yet.

Current state:

- the benchmark harness is implemented
- the four required comparison systems are implemented
- HotpotQA conversion is implemented
- supporting-fact proxy metrics are implemented
- behavior metrics are logged
- report generation is implemented

Current boundary:

- checked-in Qwen3 configs are prompt scaffolds for local verification
- paper-grade Architecture 1 numbers should only be reported from a config whose `paper_claim_blockers` list is empty

When runs are generated, render them from:

- [committee_llm/benchmark_report.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/benchmark_report.py)

into this file.
