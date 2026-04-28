# Architecture 1 Workspace

This workspace implements the paper's Architecture 1 control condition inside [implementation](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation), keeps benchmark assets in [evals](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/evals), and stores research-facing notes in [docs](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs).

Read these in order:

- [ARCHITECTURE_1_COMPREHENSIVE.md](/Users/luthiraa/Documents/korg2.4/arch-1/docs/ARCHITECTURE_1_COMPREHENSIVE.md)
- [ARCHITECTURE_1.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/ARCHITECTURE_1.md)
- [BENCHMARKS.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/BENCHMARKS.md)
- [REMOTE_EVAL_PLAN.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/REMOTE_EVAL_PLAN.md)
- [BENCHMARK_RESULTS.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/BENCHMARK_RESULTS.md)

Implementation entry points:

- [committee_llm/orchestrator.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/orchestrator.py)
- [committee_llm/baselines.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/baselines.py)
- [committee_llm/benchmark.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/benchmark.py)
- [committee_llm/evaluation.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/evaluation.py)
- [configs/qwen3_arch1_posttrained_template.json](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/configs/qwen3_arch1_posttrained_template.json)
- [configs/qwen3_8b_ollama_eval.json](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/configs/qwen3_8b_ollama_eval.json)

Current boundary:

- The code now enforces Architecture 1's fixed topology, role schemas, local context views, per-role calls, and benchmark controls.
- The checked-in runnable Qwen3 configs are prompt scaffolds for local development.
- Paper-grade Architecture 1 claims require four distinct role-tuned Qwen3 checkpoints, which are represented in the template config but not shipped in this repo.
