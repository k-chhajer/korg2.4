# Implementation Plan: Grok 4.2-Style Native Multi-Agent System

## 1) Scope and Objective

Build a research-grade, reproducible approximation of the described Grok 4.2 architecture:
- One shared base model (Qwen3 family) with role-conditioned behavior.
- Four runtime roles: `captain`, `harper`, `benjamin`, `lucas`.
- Shared-prefix inference with KV reuse.
- Short structured critique exchange.
- Joint multi-agent RL (AT-GRPO style) for role specialization and coordination.

Primary research claim to test:
- Jointly trained heterogeneous multi-agent collaboration beats or matches larger single-agent baselines at similar compute/active-parameter budget.

## 2) Assumptions and Constraints

- This is a reverse-engineering effort, not a claim of exact internal parity with xAI.
- Initial target is a buildable small-to-mid scale system (8B/14B class), not frontier pretraining scale.
- Compute and engineering constraints require staged rollout:
  1. Prompt-only multi-agent baseline.
  2. SFT warm-start.
  3. Joint multi-agent RL.
  4. Optional role-specific LoRA adapters.

## 3) System Architecture (v1)

### 3.1 Core Runtime Graph

1. `captain_plan` turn:
- Decompose task.
- Define sub-asks for each role.

2. Parallel role turns:
- `harper`: retrieval/fact grounding.
- `benjamin`: logic/math/code reasoning.
- `lucas`: alternatives, UX framing, creative variants.

3. Compressed critique round:
- Each role emits structured critique objects (not free-form debate).

4. `captain_synthesis` turn:
- Resolve conflicts.
- Produce final response with evidence-aware integration.

### 3.2 Critique Schema

Fixed JSON schema:

```json
{
  "challenge_type": "factual|logical|missing_consideration|alternative_approach",
  "target": "<=20 tokens claim span",
  "correction": "<=50 tokens correction/proposal"
}
```

Validation rules:
- Hard token caps.
- `target` must align to an existing claim span in prior outputs.
- Reject malformed critiques and request regeneration once.

### 3.3 Serving and Caching

- Inference engine: vLLM with prefix caching enabled.
- Shared prompt prefix:
  - Task context.
  - Conversation history.
  - Canonical protocol instructions.
- Role divergence starts after role token/system segment.

## 4) Repository and Components

Recommended project layout:

```text
grok_multiagent/
  configs/
    model/
    sft/
    rl/
    eval/
  data/
    raw/
    synthetic/
    processed/
  prompts/
    protocol/
    roles/
  src/
    orchestrator/
    critique/
    rewards/
    training/
    eval/
    tools/
  scripts/
    build_dataset.py
    run_sft.py
    run_rl.py
    run_eval.py
  reports/
    ablations/
    benchmark_runs/
```

## 5) Milestone Plan

## Milestone 0: Spec Freeze and Reproducibility Setup (1 week)

Deliverables:
- Frozen protocol spec (`roles`, turn order, critique schema, max tokens).
- Experiment registry format (YAML + run IDs).
- Seed policy and deterministic eval harness.

Exit criteria:
- Same config reproduces identical orchestration behavior on a fixed prompt set.

## Milestone 1: Prompt-Only Native Multi-Agent Baseline (1-2 weeks)

Goals:
- Implement runtime graph end-to-end.
- Validate that role outputs are behaviorally distinct before training.

Tasks:
- Implement orchestrator state machine.
- Add role prompt templates and captain synthesis template.
- Add critique validator and retry logic.
- Add logging for per-turn token use, latency, critique acceptance rate.

Metrics:
- Task success on a 200-item internal suite.
- Role distinctiveness score (embedding distance + rubric).
- Latency overhead vs single-agent baseline.

Exit criteria:
- Stable multi-agent pipeline in production-like serving loop.

## Milestone 2: Synthetic Trace Generation + SFT Warm Start (2-3 weeks)

Goals:
- Generate high-quality multi-agent traces and teach protocol adherence.

Dataset plan:
- 1,000 base problems across math, coding, factual, analytical tasks.
- ~5 traces/problem variants => ~5,000 trajectories.
- Store full DAG:
  - captain decomposition
  - three role outputs
  - critiques
  - final synthesis

Data quality filters:
- Format-valid traces only.
- Automatic rejection for contradiction, low evidence density, schema failure.
- Diversity filter to avoid near-duplicate trajectories.

SFT targets:
- Train role-conditioned behavior using one shared base model.
- Include special role/control tokens in input format.

Metrics:
- Protocol compliance rate.
- Role-style separability.
- Baseline benchmark deltas vs prompt-only system.

Exit criteria:
- SFT model is strictly better than prompt-only baseline on at least 3/4 benchmark categories.

## Milestone 3: Reward Layer and Verifiers (2-4 weeks)

Goals:
- Build role-conditioned reward stack with maximum verifiability.

Reward design:
- `benjamin`: RLVR-heavy.
  - Code: unit tests / pass@k.
  - Math: symbolic checks where possible.
  - Logic: consistency and contradiction checks.
- `harper`:
  - Claim extraction -> evidence retrieval -> support score.
  - Reward on supported-claim ratio and contradiction penalty.
- `lucas`:
  - Diversity reward (embedding distance from captain/benjamin drafts).
  - Optional preference model later.
- `captain`:
  - Final-task metric reward + consistency bonus.

Reward normalization:
- Per-role z-score normalization.
- Reward clipping and per-role temperature.

Exit criteria:
- Reward signals are stable (non-collapsing) over pilot rollouts.
- Inter-rater agreement of automated rewards vs human spot-check >= target threshold (define before run).

## Milestone 4: Joint Multi-Agent RL (AT-GRPO) with LoRA (3-6 weeks)

Goals:
- Train role-conditioned collaboration policy jointly.

Training setup:
- Base: Qwen3-8B first; 14B optional scale-up.
- Adaptation: LoRA (initial rank 64, tune 32-128).
- Episode structure:
  - Turn 1 captain plan.
  - Turn 2-4 role outputs (parallel generated, ordered for training record).
  - Critique exchange.
  - Captain synthesis.
- Advantage:
  - Agent-and-turn-wise grouping.

Stability controls:
- KL penalty to SFT reference.
- Curriculum:
  1. Easy deterministic tasks.
  2. Mixed tasks.
  3. Hard open-ended tasks.
- Early stop on reward hacking indicators.

Exit criteria:
- Joint RL model outperforms SFT baseline on composite score with statistically significant gains.
- No major regression in factuality or code correctness.

## Milestone 5: Role-Specialization Branch (Optional) (2-3 weeks)

Trigger:
- If joint shared-policy model shows weak role separation.

Approach:
- Keep one base model.
- Attach per-role LoRA adapters.
- Evaluate:
  - Shared-only vs per-role adapters.
  - Quality/latency/memory tradeoff.

Exit criteria:
- Adopt branch only if specialization gain exceeds defined threshold for acceptable serving overhead.

## Milestone 6: Benchmarking, Ablations, and Publication Package (2-4 weeks)

Core benchmarks:
- MATH-500
- GPQA Diamond
- LiveCodeBench
- GAIA (tool-augmented track)

Mandatory ablations:
- No critique vs structured critique.
- Prompt-only vs SFT vs joint RL.
- Joint RL vs independently trained single-role policies.
- Shared policy vs role LoRA adapters.

Primary plots:
- Quality vs active parameters.
- Quality vs latency.
- Critique token budget vs quality.

Exit criteria:
- Reproducible tables and scripts.
- Clear conclusion on whether joint heterogeneous MARL beats matched single-agent baselines.

## 6) Experiment Matrix (Minimum)

Run IDs:
- `A0`: single-agent base model.
- `A1`: prompt-only 4-agent orchestration.
- `A2`: SFT multi-agent.
- `A3`: joint RL multi-agent (shared policy).
- `A4`: joint RL + per-role LoRA.
- `A5`: independently trained role models (control).

All runs must record:
- Seed
- Model checkpoint hash
- Prompt/protocol version
- Reward config hash
- Tooling version (retriever/verifier)

## 7) Key Risks and Mitigations

1. Reward hacking / proxy overfitting
- Mitigation: hidden eval sets, adversarial tests, periodic human audits.

2. Role collapse (all roles converge to similar behavior)
- Mitigation: role-conditioned rewards, diversity constraints, adapter branch fallback.

3. Critique degeneracy (generic critiques)
- Mitigation: span-alignment checks, utility reward based on synthesis improvement.

4. Retrieval noise harms Harper
- Mitigation: source quality weighting, contradiction penalties, claim-level confidence thresholds.

5. Training instability
- Mitigation: KL anchoring, staged curriculum, smaller pilot runs before long jobs.

## 8) Concrete 30-60-90 Day Plan

## Day 0-30
- Build orchestrator + schema validation.
- Launch prompt-only baseline.
- Generate first synthetic dataset tranche.
- Train SFT warm start.

## Day 31-60
- Implement reward pipeline and verifiers.
- Run pilot AT-GRPO training.
- Debug role collapse and critique quality.
- First ablations (no critique, no Harper retrieval).

## Day 61-90
- Full joint RL runs.
- Optional role-LoRA branch.
- Benchmark suite and publication-grade figures.
- Freeze best checkpoint + reproducibility bundle.

## 9) Immediate Next Actions (This Week)

1. Freeze protocol spec in `prompts/protocol/` and `configs/`.
2. Implement orchestrator skeleton and structured critique validator.
3. Create dataset schema and generation script for synthetic traces.
4. Stand up baseline eval harness with 100-200 curated tasks.
5. Run `A1` (prompt-only) as first hard baseline before any training.

## 10) Success Criteria

Technical:
- Stable native multi-agent runtime with role-distinct outputs and bounded critique budget.

Model:
- Joint RL improves over SFT and prompt-only across heterogeneous benchmarks.

Research:
- Demonstrated quality/compute advantage or clear boundary conditions where this architecture helps.

