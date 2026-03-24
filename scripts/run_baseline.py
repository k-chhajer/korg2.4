from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator.backend import build_backend
from src.orchestrator.pipeline import MultiAgentOrchestrator
from src.orchestrator.types import ProtocolConfig, RuntimeConfig
from src.tools.run_logger import write_json
from src.tools.web_search import build_search_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-only multi-agent baseline (A1).")
    parser.add_argument(
        "--query",
        default="Create an implementation plan for a multi-agent reasoning system.",
        help="User query to run through orchestration pipeline.",
    )
    parser.add_argument("--task-id", default="manual_task", help="Task id for run trace.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic run seed.")
    parser.add_argument(
        "--runtime-config",
        default=str(ROOT / "configs" / "model" / "runtime.yaml"),
        help="Path to runtime backend/search config.",
    )
    parser.add_argument("--backend", default=None, help="Override backend provider.")
    parser.add_argument("--model", default=None, help="Override model identifier.")
    parser.add_argument("--search-provider", default=None, help="Override search provider.")
    args = parser.parse_args()

    config = ProtocolConfig.from_file(ROOT / "configs" / "protocol" / "protocol.yaml")
    runtime_config = RuntimeConfig.from_file(Path(args.runtime_config)).with_overrides(
        backend_provider=args.backend,
        model=args.model,
        search_provider=args.search_provider,
    )
    backend = build_backend(runtime_config)
    search_tool = build_search_tool(runtime_config)
    orchestrator = MultiAgentOrchestrator(
        root=ROOT,
        config=config,
        runtime_config=runtime_config,
        backend=backend,
        search_tool=search_tool,
    )

    record = orchestrator.run(
        run_id="A1",
        task_id=args.task_id,
        user_query=args.query,
        seed=args.seed,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "reports" / "benchmark_runs" / f"a1_baseline_{ts}.json"
    payload = {
        **asdict(record),
        "runtime": {
            "backend_provider": runtime_config.backend.get("provider"),
            "model": runtime_config.backend.get("model"),
            "search_provider": runtime_config.search.get("provider"),
        },
    }
    write_json(out_path, payload)

    print("Run complete.")
    print(f"Output file: {out_path}")
    print("Runtime:")
    print(
        json.dumps(
            {
                "backend_provider": runtime_config.backend.get("provider"),
                "model": runtime_config.backend.get("model"),
                "search_provider": runtime_config.search.get("provider"),
            },
            indent=2,
        )
    )
    print("Metrics:")
    print(json.dumps(record.metrics, indent=2))
    print("\nFinal answer:\n")
    print(record.final_answer)


if __name__ == "__main__":
    main()

