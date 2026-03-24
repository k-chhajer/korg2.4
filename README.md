# korg2.4
reverse engineering grok 4.2

Current Milestone 1 runtime:
- captain decomposes the task
- Harper proposes search queries and consumes web results
- Benjamin performs rigorous reasoning
- Lucas reviews Harper and Benjamin, then critiques
- captain synthesizes the final answer

Usage:

```powershell
python scripts/run_baseline.py
python scripts/run_baseline.py --backend openrouter --model qwen/qwen3-14b --search-provider duckduckgo
python scripts/run_eval.py
```

Environment variables for live runs:
- `OPENROUTER_API_KEY` for Qwen or other hosted models via OpenRouter
- `SERPER_API_KEY` if you choose `--search-provider serper`

Notes:
- `duckduckgo` works as a best-effort no-key fallback for search, but it is less reliable than a provider API.
- The default runtime config lives at `configs/model/runtime.yaml`.
- Replace `qwen/qwen3-14b` with the exact OpenRouter Qwen model ID you have access to if it differs.
