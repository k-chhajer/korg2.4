# Remote Eval Plan

This plan follows the benchmark instructions for Architecture 1 as the external-specialization control and keeps the run order compute-conscious.

## Scope

Run these systems:

- `single_shot`
- `single_multiturn`
- `single_structured`
- `committee`

Main benchmark:

- HotpotQA, `100` examples

Secondary benchmark:

- FRAMES, `75` examples, only if the pipeline performs real retrieval

Stress benchmark:

- custom stress set, `50` examples

Main metrics:

- answer accuracy
- answer F1 where applicable
- HotpotQA supporting-fact proxy metrics
- input tokens
- output tokens
- total tokens
- model calls
- wall-clock latency

Behavior metrics:

- decomposition rate
- evidence-grounding rate
- revision-after-critique rate
- premature-finalization rate
- unsupported-claim rate

Required ablations:

- no critic
- no researcher
- no analyst
- order swap: coordinator -> analyst -> researcher -> critic -> coordinator
- order swap: coordinator -> researcher -> critic -> analyst -> coordinator
- budget-matched single-model baseline

## Architecture 1 Boundary

Architecture 1 is one composite system externally:

- one prompt in
- one final answer out

Internally it is a fixed committee:

- coordinator -> researcher -> analyst -> critic -> coordinator

Paper-valid Architecture 1 results require:

- `specialization_source = post_trained_role_specialists`
- distinct role checkpoints
- no checkpoint reuse across roles

If `paper_claim_blockers` is non-empty, the run is a scaffold run and must not be presented as a post-trained committee result.

## Recommended Modal Order

Run in this order:

1. deploy one remote Qwen3-8B vLLM endpoint on Modal
2. verify one committee sample task
3. run `10` HotpotQA examples on all 4 systems
4. if stable, run `100` HotpotQA examples on all 4 systems
5. run `50` custom stress examples on all 4 systems
6. run `75` FRAMES examples only if retrieval is actually enabled
7. run ablations on:
   - `40` HotpotQA
   - `20` custom stress
8. render the reports

Do not start with FRAMES.
Do not start with ablations.
Do not spend budget on Architecture 1 before the main comparison table is complete.

## Budget Guidance

Using current Modal pricing references for `T4` and `L4`, the practical recommendation is:

- start on `T4`
- use `Qwen3 8B`
- keep `30B` out of the first pass

Estimated cost bands:

- `100 HotpotQA` on all 4 systems: about `$3-$8` on `T4`
- `50` custom stress on all 4 systems: about `$1-$4` on `T4`
- `75` FRAMES on all 4 systems: about `$2-$7` on `T4`
- ablations on reduced subsets: about `$6-$12` additional on `T4`

Expected total:

- main suite without FRAMES: about `$4-$12`
- main suite with FRAMES: about `$6-$19`
- main suite plus ablations: about `$12-$30`

This is why the safest sequence is:

1. HotpotQA
2. custom stress
3. FRAMES only if needed
4. ablations last

## Exact Run Structure

### Stage 1: Endpoint Verification

Goal:

- prove the remote endpoint is stable and OpenAI-compatible

Run:

- one sample task through `committee`
- one HotpotQA example through `single_shot`

Pass criteria:

- traces write successfully
- no schema failures in committee stages
- no backend timeout

### Stage 2: Small Smoke Benchmark

Goal:

- validate all 4 systems before the full run

Run:

- `10` HotpotQA examples
- all 4 systems

Check:

- summary renders cleanly
- all systems complete
- token and latency accounting looks sane

### Stage 3: Main Benchmark

Run:

- `100` HotpotQA examples
- all 4 systems

Report:

- one main results table
- one compute table
- one behavior table

Then run:

- `50` custom stress examples
- all 4 systems

Only then decide whether FRAMES is justified.

### Stage 4: FRAMES Decision

Run FRAMES only if the actual system does retrieval or live evidence gathering beyond provided context.

If yes:

- run `75` FRAMES examples
- all 4 systems

If no:

- omit FRAMES from the paper
- state that FRAMES was excluded because retrieval is not an active pipeline component

### Stage 5: Ablations

Run on reduced subsets:

- `40` HotpotQA
- `20` custom stress

Run:

- full committee
- no critic
- no researcher
- no analyst
- order swap 1
- order swap 2
- budget-matched single-model baseline

## Output Files

Keep runs under:

- [evals/runs](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/evals/runs)

Render benchmark reports into:

- [BENCHMARK_RESULTS.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/BENCHMARK_RESULTS.md)

Use these docs alongside:

- [ARCHITECTURE_1.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/ARCHITECTURE_1.md)
- [BENCHMARKS.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/BENCHMARKS.md)
- [IMPLEMENTATION_NOTE.md](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/docs/IMPLEMENTATION_NOTE.md)

## Paper Framing

Use this framing:

- Architecture 1 is a single end-to-end composite system from the user perspective
- internally it is a fixed committee of role-specialized model instances
- it is the external-specialization control, not the novelty claim
- the unified system is the main object of comparison
