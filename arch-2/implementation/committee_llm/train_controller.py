from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from committee_llm.adaptive import (
    ACTION_NAMES,
    AdaptiveCommitteeRunner,
    AdaptiveRuntimeConfig,
    HeuristicControllerPolicy,
    N_PHASES,
    REASONING_PHASES,
    RewardConfig,
    SEMANTIC_EMBED_DIM,
    TaskSpec,
    TASK_PROGRESS_DIM,
    TorchControllerPolicy,
)
from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - depends on runtime
    raise RuntimeError(
        "Torch is required for controller training. Install it in Colab with `pip install torch`."
    ) from exc


class ReasoningAwareController(nn.Module):
    """
    Controller with a GRU hidden state that tracks reasoning progress.
    """

    def __init__(
        self,
        state_dim: int = 28,
        n_phases: int = 6,
        n_actions: int = 6,
        hidden_size: int = 128,
        semantic_dim: int = SEMANTIC_EMBED_DIM,
        semantic_proj_dim: int = 32,
        semantic_delta_dim: int = SEMANTIC_EMBED_DIM,
        semantic_delta_proj_dim: int = 16,
        task_progress_dim: int = TASK_PROGRESS_DIM,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim
        self.semantic_delta_dim = semantic_delta_dim
        self.semantic_delta_proj_dim = semantic_delta_proj_dim
        self.task_progress_dim = task_progress_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, semantic_proj_dim),
        )
        self.semantic_delta_proj = nn.Sequential(
            nn.Linear(semantic_delta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, semantic_delta_proj_dim),
        )
        gru_input_dim = state_dim + n_phases + semantic_proj_dim + semantic_delta_proj_dim + task_progress_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_size)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.phase_probe = nn.Linear(hidden_size, n_phases)
        self.hidden_size = hidden_size

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size)

    def forward(
        self,
        state_features: torch.Tensor,
        phase_onehot: torch.Tensor,
        semantic_embed: torch.Tensor | None = None,
        task_progress: torch.Tensor | None = None,
        h_prev: torch.Tensor | None = None,
        semantic_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            h_prev is None
            and task_progress is None
            and semantic_embed is not None
            and semantic_embed.shape[-1] == self.hidden_size
        ):
            h_prev = semantic_embed
            semantic_embed = None
        if h_prev is None:
            h_prev = self.initial_hidden(batch_size=state_features.shape[0]).to(state_features.device)
        if semantic_embed is None:
            semantic_embed = torch.zeros(
                state_features.shape[0], self.semantic_dim, dtype=state_features.dtype, device=state_features.device
            )
        if task_progress is None:
            task_progress = torch.zeros(
                state_features.shape[0],
                self.task_progress_dim,
                dtype=state_features.dtype,
                device=state_features.device,
            )
        if semantic_delta is None:
            semantic_delta = torch.zeros(
                state_features.shape[0],
                self.semantic_delta_dim,
                dtype=state_features.dtype,
                device=state_features.device,
            )
        semantic = self.semantic_proj(semantic_embed)
        delta = self.semantic_delta_proj(semantic_delta)
        x = torch.cat([state_features, phase_onehot, semantic, delta, task_progress], dim=-1)
        h_t = self.gru(x, h_prev)
        action_logits = self.policy_head(h_t)
        value = self.value_head(h_t).squeeze(-1)
        phase_logits = self.phase_probe(h_t)
        return action_logits, value, h_t, phase_logits


def _load_tasks(tasks_path: str | Path, limit: int | None = None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_number, raw_line in enumerate(Path(tasks_path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Task line {line_number} in {tasks_path} must be a JSON object.")
        tasks.append(TaskSpec.from_dict(payload, default_task_id=f"task-{line_number:05d}"))
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError(f"No tasks loaded from {tasks_path}.")
    return tasks


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _init_or_load_rollout_dump(
    *,
    path: Path,
    args: argparse.Namespace,
    config: ExperimentConfig,
    device: torch.device,
    task_count: int,
    train_task_count: int,
    eval_task_count: int,
) -> dict[str, Any]:
    if path.exists():
        try:
            payload = _load_json(path)
            if isinstance(payload, dict):
                payload.setdefault("schema_version", 1)
                payload.setdefault("run_info", {})
                payload.setdefault("train_episodes", [])
                payload.setdefault("eval_events", [])
                return payload
        except Exception:
            pass

    payload = {
        "schema_version": 1,
        "run_info": {
            "args": vars(args),
            "device": str(device),
            "task_count": task_count,
            "train_task_count": train_task_count,
            "eval_task_count": eval_task_count,
            "config_name": config.name,
            "config_path": str(config.source_path),
        },
        "train_episodes": [],
        "eval_events": [],
    }
    _write_json(path, payload)
    return payload


def _truncate_rollout_dump_for_resume(payload: dict[str, Any], *, start_episode: int) -> dict[str, Any]:
    train_episodes = payload.get("train_episodes", [])
    eval_events = payload.get("eval_events", [])
    if isinstance(train_episodes, list):
        payload["train_episodes"] = [
            item for item in train_episodes if int(item.get("episode", 0)) <= int(start_episode)
        ]
    else:
        payload["train_episodes"] = []
    if isinstance(eval_events, list):
        payload["eval_events"] = [
            item for item in eval_events if int(item.get("episode", 0)) <= int(start_episode)
        ]
    else:
        payload["eval_events"] = []
    return payload


def _split_tasks(tasks: list[TaskSpec], eval_count: int, seed: int) -> tuple[list[TaskSpec], list[TaskSpec]]:
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return list(shuffled), list(shuffled)
    eval_count = max(1, min(eval_count, len(shuffled) - 1))
    eval_tasks = shuffled[:eval_count]
    train_tasks = shuffled[eval_count:]
    return train_tasks, eval_tasks


def _trajectory_from_rollout(rollout: Any) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for transition in rollout.transitions:
        if transition.log_prob is None or transition.action_logits is None:
            continue
        step_payload = {
            "log_prob": transition.log_prob,
            "action_logits": transition.action_logits,
            "phase_logits": transition.phase_logits,
            "phase_label": transition.phase_label,
        }
        steps.append(step_payload)

    controller = rollout.trace.get("controller", {})
    return {
        "steps": steps,
        "final_answer": rollout.trace.get("final_answer", ""),
        "action_sequence": list(controller.get("action_sequence", [])),
        "phase_sequence": list(controller.get("phase_sequence", [])),
        "reward": float(rollout.total_reward),
        "quality_score": float(rollout.quality_score),
        "trace": rollout.trace,
    }


def _compute_grpo_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    if len(rewards) < 2:
        return [reward - 0.5 for reward in rewards]
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)) ** 0.5
    std_reward = max(std_reward, 1e-8)
    return [(reward - mean_reward) / std_reward for reward in rewards]


def grpo_loss(
    trajectories: list[dict[str, Any]],
    rewards: list[float],
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    Group Relative Policy Optimization.
    """
    advantages = _compute_grpo_advantages(rewards)

    first_log_prob = None
    for trajectory in trajectories:
        for step in trajectory.get("steps", []):
            if step.get("log_prob") is not None:
                first_log_prob = step["log_prob"]
                break
        if first_log_prob is not None:
            break

    if first_log_prob is None:
        return torch.tensor(0.0)

    device = first_log_prob.device
    policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy_loss = torch.zeros((), dtype=torch.float32, device=device)

    for trajectory, advantage in zip(trajectories, advantages):
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=device)
        for step in trajectory.get("steps", []):
            log_prob = step.get("log_prob")
            action_logits = step.get("action_logits")
            if log_prob is None or action_logits is None:
                continue
            policy_loss = policy_loss - log_prob * advantage_tensor
            probs = torch.softmax(action_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropy_loss = entropy_loss - entropy_coef * entropy

    return policy_loss + entropy_loss


def auxiliary_phase_loss(
    trajectories: list[dict[str, Any]],
    phase_loss_weight: float = 0.1,
) -> torch.Tensor:
    """
    Auxiliary cross-entropy loss on the phase probe head.
    """
    first_phase_logits = None
    for trajectory in trajectories:
        for step in trajectory.get("steps", []):
            if step.get("phase_logits") is not None:
                first_phase_logits = step["phase_logits"]
                break
        if first_phase_logits is not None:
            break

    if first_phase_logits is None:
        return torch.tensor(0.0)

    device = first_phase_logits.device
    criterion = nn.CrossEntropyLoss()
    loss = torch.zeros((), dtype=torch.float32, device=device)

    for trajectory in trajectories:
        for step in trajectory.get("steps", []):
            phase_logits = step.get("phase_logits")
            phase_label = step.get("phase_label")
            if phase_logits is None or phase_label is None:
                continue
            phase_label = phase_label.to(device)
            loss = loss + criterion(phase_logits, phase_label)

    return phase_loss_weight * loss


def _evaluate_policy(
    *,
    runner: AdaptiveCommitteeRunner,
    policy: TorchControllerPolicy,
    tasks: list[TaskSpec],
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = tasks[: max(1, min(limit, len(tasks)))]
    rewards: list[float] = []
    qualities: list[float] = []
    tokens: list[float] = []
    elapsed: list[float] = []
    decisions: list[float] = []
    phase_sequences: list[list[str]] = []
    action_sequences: list[list[str]] = []
    rollouts_payload: list[dict[str, Any]] = []

    with torch.no_grad():
        for task in selected:
            rollout = runner.rollout_task(task, policy, deterministic=True, track_grad=False)
            controller = rollout.trace["controller"]
            rewards.append(float(rollout.total_reward))
            qualities.append(float(rollout.quality_score))
            tokens.append(float(controller["used_tokens"]))
            elapsed.append(float(rollout.trace["total_elapsed_sec"]))
            decisions.append(float(controller["decision_count"]))
            phase_sequence = list(controller.get("phase_sequence", []))
            action_sequence = list(controller.get("action_sequence", []))
            phase_sequences.append(phase_sequence)
            action_sequences.append(action_sequence)
            rollouts_payload.append(
                {
                    "task_id": task.task_id,
                    "benchmark_name": task.benchmark_name,
                    "quality_score": float(rollout.quality_score),
                    "total_reward": float(rollout.total_reward),
                    "phase_sequence": phase_sequence,
                    "action_sequence": action_sequence,
                    "trace": rollout.trace,
                }
            )

    count = float(len(selected))
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    metrics = {
        "count": count,
        "mean_reward": sum(rewards) / count,
        "mean_quality": sum(qualities) / count,
        "mean_tokens": sum(tokens) / count,
        "mean_elapsed_sec": sum(elapsed) / count,
        "mean_decisions": sum(decisions) / count,
        "group_rewards": rewards,
        "group_reward_mean": sum(rewards) / count,
        "group_reward_std": reward_std,
        "phase_sequence": phase_sequences[0] if phase_sequences else [],
        "actions": action_sequences[0] if action_sequences else [],
    }
    return metrics, rollouts_payload


def _save_checkpoint(
    *,
    path: Path,
    episode: int,
    model: Any,
    optimizer: torch.optim.Optimizer,
    best_eval_reward: float,
    train_args: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_eval_reward": best_eval_reward,
            "train_args": train_args,
        },
        str(path),
    )


def _load_checkpoint(path: Path, model: Any, optimizer: torch.optim.Optimizer) -> tuple[int, float]:
    payload = torch.load(str(path), map_location="cpu")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    episode = int(payload.get("episode", 0))
    best_eval_reward = float(payload.get("best_eval_reward", float("-inf")))
    return episode, best_eval_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an Architecture 2 adaptive controller on top of frozen committee specialists."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config JSON.")
    parser.add_argument("--tasks", required=True, help="Path to benchmark task JSONL.")
    parser.add_argument("--outdir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--limit", type=int, help="Optional max number of tasks to load.")
    parser.add_argument("--eval-task-count", type=int, default=24, help="Number of held-out tasks for evaluation.")
    parser.add_argument("--episodes", type=int, default=120, help="Training episodes.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N episodes.")
    parser.add_argument("--save-every", type=int, default=5, help="Write checkpoint every N episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grpo-group-size", type=int, default=4,
                        help="Number of trajectories to sample per question for GRPO normalization")
    parser.add_argument("--phase-loss-weight", type=float, default=0.1,
                        help="Weight for auxiliary phase classification loss")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient for exploration")
    parser.add_argument("--cheap-model", type=str, default="anthropic/claude-haiku-4-5-20251001",
                        help="Model used for phase classification (should be cheapest available)")
    parser.add_argument("--semantic-model", type=str, default="all-MiniLM-L6-v2",
                        help="Local sentence-transformers model for frozen semantic state embeddings")
    parser.add_argument("--disable-semantic-embeddings", action="store_true",
                        help="Disable local semantic embeddings and feed zeros instead")

    parser.add_argument("--budget-tokens", type=int, default=16000)
    parser.add_argument("--max-decisions", type=int, default=6)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--per-role-call-cap", type=int, default=2)
    parser.add_argument("--min-tokens-for-call", type=int, default=300)

    parser.add_argument("--device", default="auto", help="Torch device: auto|cpu|cuda")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and run deterministic eval.")
    parser.add_argument(
        "--warmstart-heuristic",
        action="store_true",
        help="Run one deterministic heuristic episode first to verify backend and prompts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = ExperimentConfig.load(args.config)
    tasks = _load_tasks(args.tasks, limit=args.limit)
    train_tasks, eval_tasks = _split_tasks(tasks, eval_count=args.eval_task_count, seed=args.seed)

    client = OpenAICompatibleChatClient(
        api_base=config.provider.api_base,
        api_key_env=config.provider.api_key_env,
        timeout_sec=config.provider.timeout_sec,
    )

    runtime = AdaptiveRuntimeConfig(
        budget_tokens=args.budget_tokens,
        max_decisions=args.max_decisions,
        max_restarts=args.max_restarts,
        per_role_call_cap=args.per_role_call_cap,
        min_tokens_for_call=args.min_tokens_for_call,
        cheap_model=args.cheap_model,
        semantic_model=args.semantic_model,
        use_semantic_embeddings=not args.disable_semantic_embeddings,
    )
    reward = RewardConfig()
    runner = AdaptiveCommitteeRunner(config=config, client=client, runtime=runtime, reward=reward)

    model = ReasoningAwareController(
        state_dim=runner.state_size,
        n_phases=N_PHASES,
        n_actions=len(ACTION_NAMES),
        hidden_size=args.hidden_size,
        semantic_dim=SEMANTIC_EMBED_DIM,
        semantic_proj_dim=32,
        semantic_delta_dim=SEMANTIC_EMBED_DIM,
        semantic_delta_proj_dim=16,
        task_progress_dim=TASK_PROGRESS_DIM,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    policy = TorchControllerPolicy(model=model, device=str(device))

    start_episode = 0
    best_eval_reward = float("-inf")
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if checkpoint_path.exists():
            start_episode, best_eval_reward = _load_checkpoint(checkpoint_path, model, optimizer)
            print(f"resumed from {checkpoint_path} at episode {start_episode}")

    _write_json(
        outdir / "train_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "task_count": len(tasks),
            "train_task_count": len(train_tasks),
            "eval_task_count": len(eval_tasks),
            "config_name": config.name,
            "config_path": str(config.source_path),
            "reasoning_phases": list(REASONING_PHASES),
            "semantic_embedding_dim": SEMANTIC_EMBED_DIM,
            "semantic_projection_dim": 32,
            "semantic_delta_dim": SEMANTIC_EMBED_DIM,
            "semantic_delta_projection_dim": 16,
            "task_progress_dim": TASK_PROGRESS_DIM,
        },
    )
    rollout_dump_path = outdir / "run_rollouts.json"
    rollout_dump = _init_or_load_rollout_dump(
        path=rollout_dump_path,
        args=args,
        config=config,
        device=device,
        task_count=len(tasks),
        train_task_count=len(train_tasks),
        eval_task_count=len(eval_tasks),
    )
    rollout_dump = _truncate_rollout_dump_for_resume(rollout_dump, start_episode=start_episode)
    _write_json(rollout_dump_path, rollout_dump)

    if args.warmstart_heuristic:
        warm_policy = HeuristicControllerPolicy()
        warm_rollout = runner.rollout_task(train_tasks[0], warm_policy, deterministic=True, track_grad=False)
        _write_json(outdir / "warmstart_heuristic_trace.json", warm_rollout.trace)
        print("wrote warmstart_heuristic_trace.json")

    if args.evaluate_only:
        if not args.resume:
            raise ValueError("--evaluate-only requires --resume <checkpoint_path>.")
        metrics, eval_rollouts = _evaluate_policy(
            runner=runner,
            policy=policy,
            tasks=eval_tasks,
            limit=min(args.eval_task_count, len(eval_tasks)),
        )
        metrics["phase_loss"] = None
        metrics["grpo_advantages"] = []
        rollout_dump["eval_events"].append(
            {
                "episode": int(start_episode),
                "mode": "evaluate_only",
                "metrics": metrics,
                "rollouts": eval_rollouts,
            }
        )
        _write_json(rollout_dump_path, rollout_dump)
        _write_json(outdir / "eval_only_metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        return 0

    train_log_path = outdir / "train_log.jsonl"
    eval_log_path = outdir / "eval_log.jsonl"
    latest_ckpt = outdir / "checkpoint_last.pt"
    best_ckpt = outdir / "checkpoint_best.pt"

    rng = random.Random(args.seed + 19)
    for episode in range(start_episode + 1, args.episodes + 1):
        model.train()
        task = rng.choice(train_tasks)

        trajectories: list[dict[str, Any]] = []
        rewards: list[float] = []
        rollouts: list[Any] = []
        group_size = max(1, args.grpo_group_size)

        for _ in range(group_size):
            rollout = runner.rollout_task(task, policy, deterministic=False, track_grad=True)
            trajectory = _trajectory_from_rollout(rollout)
            trajectories.append(trajectory)
            rewards.append(float(rollout.total_reward))
            rollouts.append(rollout)

        advantages = _compute_grpo_advantages(rewards)
        pg_loss = grpo_loss(trajectories, rewards, entropy_coef=args.entropy_coef)
        phase_loss = auxiliary_phase_loss(trajectories, phase_loss_weight=args.phase_loss_weight)
        total_loss = pg_loss + phase_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        best_index = max(range(len(rewards)), key=lambda idx: rewards[idx]) if rewards else 0
        best_rollout = rollouts[best_index]
        best_trace_controller = best_rollout.trace.get("controller", {})
        mean_reward = mean(rewards) if rewards else 0.0
        reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        mean_quality = mean([float(item.quality_score) for item in rollouts]) if rollouts else 0.0
        mean_tokens = mean([float(item.trace["controller"]["used_tokens"]) for item in rollouts]) if rollouts else 0.0
        mean_decisions = mean([float(item.trace["controller"]["decision_count"]) for item in rollouts]) if rollouts else 0.0
        mean_elapsed = mean([float(item.trace["total_elapsed_sec"]) for item in rollouts]) if rollouts else 0.0

        record = {
            "episode": episode,
            "task_id": task.task_id,
            "quality_score": mean_quality,
            "total_reward": mean_reward,
            "total_tokens": int(mean_tokens),
            "decision_count": int(mean_decisions),
            "elapsed_sec": float(mean_elapsed),
            "loss": float(total_loss.detach().cpu().item()),
            "policy_loss": float(pg_loss.detach().cpu().item()) if hasattr(pg_loss, "detach") else float(pg_loss),
            "phase_loss": float(phase_loss.detach().cpu().item()) if hasattr(phase_loss, "detach") else float(phase_loss),
            "grad_norm": float(grad_norm.detach().cpu().item()) if hasattr(grad_norm, "detach") else float(grad_norm),
            "actions": list(best_trace_controller.get("action_sequence", [])),
            "phase_sequence": list(best_trace_controller.get("phase_sequence", [])),
            "grpo_advantages": [float(value) for value in advantages],
            "group_rewards": [float(value) for value in rewards],
            "group_reward_mean": float(mean_reward),
            "group_reward_std": float(reward_std),
        }
        _append_jsonl(train_log_path, record)
        rollout_dump["train_episodes"].append(
            {
                "episode": episode,
                "task_id": task.task_id,
                "benchmark_name": task.benchmark_name,
                "record": record,
                "group_rollouts": [item.trace for item in rollouts],
            }
        )
        _write_json(rollout_dump_path, rollout_dump)
        print(
            f"episode={episode} mean_reward={record['total_reward']:.4f} mean_quality={record['quality_score']:.4f} "
            f"tokens={record['total_tokens']} loss={record['loss']:.4f}"
        )

        if episode % args.save_every == 0:
            _save_checkpoint(
                path=latest_ckpt,
                episode=episode,
                model=model,
                optimizer=optimizer,
                best_eval_reward=best_eval_reward,
                train_args=vars(args),
            )

        if episode % args.eval_every == 0:
            model.eval()
            eval_metrics, eval_rollouts = _evaluate_policy(
                runner=runner,
                policy=policy,
                tasks=eval_tasks,
                limit=min(args.eval_task_count, len(eval_tasks)),
            )
            eval_metrics["episode"] = episode
            eval_metrics["phase_loss"] = None
            eval_metrics["grpo_advantages"] = []
            _append_jsonl(eval_log_path, eval_metrics)
            rollout_dump["eval_events"].append(
                {
                    "episode": episode,
                    "mode": "during_training",
                    "metrics": eval_metrics,
                    "rollouts": eval_rollouts,
                }
            )
            _write_json(rollout_dump_path, rollout_dump)
            print(
                f"eval episode={episode} mean_reward={eval_metrics['mean_reward']:.4f} "
                f"mean_quality={eval_metrics['mean_quality']:.4f} "
                f"mean_tokens={eval_metrics['mean_tokens']:.1f}"
            )
            if float(eval_metrics["mean_reward"]) > best_eval_reward:
                best_eval_reward = float(eval_metrics["mean_reward"])
                _save_checkpoint(
                    path=best_ckpt,
                    episode=episode,
                    model=model,
                    optimizer=optimizer,
                    best_eval_reward=best_eval_reward,
                    train_args=vars(args),
                )
                print(f"updated best checkpoint at episode {episode}")

    _save_checkpoint(
        path=latest_ckpt,
        episode=args.episodes,
        model=model,
        optimizer=optimizer,
        best_eval_reward=best_eval_reward,
        train_args=vars(args),
    )
    print(f"training complete. latest checkpoint: {latest_ckpt}")
    if best_ckpt.exists():
        print(f"best checkpoint: {best_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
