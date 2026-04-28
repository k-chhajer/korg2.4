# Implementation Note

This update tightened Architecture 1 into a research-grade control scaffold.

What changed:

- enforced the fixed `coordinator -> researcher -> analyst -> critic -> coordinator` flow
- added per-role output schemas and local context views
- added schema validation and per-stage trace logging
- added `paper_claim_blockers` so prompt-scaffold runs cannot be confused with post-trained committee runs
- added the required baselines: `single_shot`, `single_multiturn`, `single_structured`
- added behavior metrics: decomposition, evidence grounding, revision after critique, premature finalization, unsupported claims
- added HotpotQA supporting-fact proxy metrics
- tagged Qwen3 configs with Architecture 1 metadata
- added a post-trained checkpoint template config for paper-valid runs
- rewrote the docs to separate scaffold status from paper-valid claims

Verification:

- local unit tests passed: `12/12`
- local `qwen3:8b` committee smoke run completed end to end

Current boundary:

- checked-in runnable Qwen3 configs are development scaffolds
- paper-grade Architecture 1 results require distinct role-tuned Qwen3 checkpoints
