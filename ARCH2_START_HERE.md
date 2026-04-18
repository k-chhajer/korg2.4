# Arch2 Start Here

Use only this folder:

- [arch-2](C:/Dev/grok_multiagent/arch-2)

Primary docs:

- [arch-2/README.md](C:/Dev/grok_multiagent/arch-2/README.md)
- [arch-2/docs/COLAB_T4_ARCH2_CONTROLLER_TRAINING.md](C:/Dev/grok_multiagent/arch-2/docs/COLAB_T4_ARCH2_CONTROLLER_TRAINING.md)

Primary training command (run from `arch-2/implementation`):

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
