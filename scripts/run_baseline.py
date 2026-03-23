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

from src.orchestrator.backend import MockBackend
from src.orchestrator.pipeline import MultiAgentOrchestrator
from src.orchestrator.types import ProtocolConfig
from src.tools.run_logger import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-only multi-agent baseline (A1).")
    parser.add_argument(
        "--query",
        default="Create an implementation plan for a multi-agent reasoning system.",
        help="User query to run through orchestration pipeline.",
    )
    parser.add_argument("--task-id", default="manual_task", help="Task id for run trace.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic run seed.")
    args = parser.parse_args()

    config = ProtocolConfig.from_file(ROOT / "configs" / "protocol" / "protocol.yaml")
    backend = MockBackend()
    orchestrator = MultiAgentOrchestrator(root=ROOT, config=config, backend=backend)

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
        "events": [asdict(event) for event in record.events],
    }
    write_json(out_path, payload)

    print("Run complete.")
    print(f"Output file: {out_path}")
    print("Metrics:")
    print(json.dumps(record.metrics, indent=2))
    print("\nFinal answer:\n")
    print(record.final_answer)


if __name__ == "__main__":
    main()

