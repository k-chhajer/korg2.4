from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.harness import load_jsonl, run_eval
from src.orchestrator.backend import build_backend
from src.orchestrator.pipeline import MultiAgentOrchestrator
from src.orchestrator.types import ProtocolConfig, RuntimeConfig
from src.tools.run_logger import write_json
from src.tools.web_search import build_search_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation for the multi-agent system.")
    parser.add_argument(
        "--runtime-config",
        default=str(ROOT / "configs" / "model" / "runtime.yaml"),
        help="Path to runtime backend/search config.",
    )
    parser.add_argument("--backend", default=None, help="Override backend provider.")
    parser.add_argument("--model", default=None, help="Override model identifier.")
    parser.add_argument("--search-provider", default=None, help="Override search provider.")
    args = parser.parse_args()

    eval_cfg = json.loads((ROOT / "configs" / "eval" / "eval_config.yaml").read_text(encoding="utf-8"))
    dataset_path = ROOT / eval_cfg["dataset_path"]
    tasks = load_jsonl(dataset_path)

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

    report = run_eval(orchestrator=orchestrator, tasks=tasks, run_id="A1", seed=314159)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / eval_cfg["report_dir"] / f"a1_eval_report_{ts}.json"
    write_json(
        out_path,
        {
            **report,
            "runtime": {
                "backend_provider": runtime_config.backend.get("provider"),
                "model": runtime_config.backend.get("model"),
                "search_provider": runtime_config.search.get("provider"),
            },
        },
    )

    print("Eval complete.")
    print(f"Report file: {out_path}")
    print(
        json.dumps(
            {
                "backend_provider": runtime_config.backend.get("provider"),
                "model": runtime_config.backend.get("model"),
                "search_provider": runtime_config.search.get("provider"),
                "task_count": report["task_count"],
                "mean_keyword_score": report["mean_keyword_score"],
                "mean_role_distinctiveness": report["mean_role_distinctiveness"],
                "mean_critique_acceptance_rate": report["mean_critique_acceptance_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

