from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.harness import load_jsonl, run_eval
from src.orchestrator.backend import MockBackend
from src.orchestrator.pipeline import MultiAgentOrchestrator
from src.orchestrator.types import ProtocolConfig
from src.tools.run_logger import write_json


def main() -> None:
    eval_cfg = json.loads((ROOT / "configs" / "eval" / "eval_config.yaml").read_text(encoding="utf-8"))
    dataset_path = ROOT / eval_cfg["dataset_path"]
    tasks = load_jsonl(dataset_path)

    config = ProtocolConfig.from_file(ROOT / "configs" / "protocol" / "protocol.yaml")
    backend = MockBackend()
    orchestrator = MultiAgentOrchestrator(root=ROOT, config=config, backend=backend)

    report = run_eval(orchestrator=orchestrator, tasks=tasks, run_id="A1", seed=314159)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / eval_cfg["report_dir"] / f"a1_eval_report_{ts}.json"
    write_json(out_path, report)

    print("Eval complete.")
    print(f"Report file: {out_path}")
    print(
        json.dumps(
            {
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

