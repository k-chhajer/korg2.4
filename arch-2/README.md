# Architecture 2 Workspace

This folder is fully separate from `arch-1` and is the only place you need for Architecture 2 work.

Core idea:

- frozen specialist committee (coordinator/researcher/analyst/critic)
- reasoning-aware learned controller decides which specialist to call next under a token budget
- controller uses a GRU hidden state + phase classifier
- training uses GRPO with verifiable reward (RLVR)
- only the controller is trained; specialist prompts/models remain frozen

Use these paths:

- training code: `implementation/committee_llm/train_controller.py`
- adaptive runtime: `implementation/committee_llm/adaptive.py`
- frozen specialist config: `implementation/configs/qwen3_8b_openrouter_arch2_controller.json`
- benchmark tasks: `evals/data/benchmarks/`
- colab runbook: `docs/COLAB_T4_ARCH2_CONTROLLER_TRAINING.md`
- OpenRouter config: `implementation/configs/qwen3_8b_openrouter_arch2_controller.json`

Quick start (Colab):

1. Set `OPENROUTER_API_KEY` in your Colab runtime.
2. `cd /content/grok_multiagent/arch-2/implementation`
3. Run:

```bash
OPENROUTER_API_KEY=your_key_here python3.11 -m committee_llm.train_controller \
  --config /content/grok_multiagent/arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json \
  --tasks /content/grok_multiagent/arch-2/evals/data/benchmarks/hotpotqa_dev.jsonl \
  --outdir /content/drive/MyDrive/korg2_4_runs/arch2_controller_hotpot \
  --episodes 120 \
  --eval-every 10 \
  --save-every 5 \
  --eval-task-count 24 \
  --grpo-group-size 4 \
  --phase-loss-weight 0.1 \
  --entropy-coef 0.01 \
  --cheap-model anthropic/claude-haiku-4-5-20251001 \
  --budget-tokens 16000 \
  --max-decisions 6 \
  --max-restarts 1 \
  --per-role-call-cap 2 \
  --device cuda
```

Artifacts produced in `--outdir`:

- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `train_log.jsonl` (includes `phase_sequence`, `phase_loss`, GRPO group stats)
- `eval_log.jsonl` (includes phase/action sequence summaries)
- `train_config.json`
- `run_rollouts.json`
