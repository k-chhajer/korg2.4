from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from committee_llm.adaptive import (
    AdaptiveCommitteeRunner,
    AdaptiveRuntimeConfig,
    HeuristicControllerPolicy,
    RewardConfig,
    TaskSpec,
    TorchControllerModel,
    TorchControllerPolicy,
)
from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - depends on runtime
    raise RuntimeError(
        "Torch is required for controller training. Install it in Colab with `pip install torch`."
    ) from exc


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


def _build_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return torch.tensor(returns, dtype=torch.float32, device=device)


def _train_step(
    *,
    optimizer: torch.optim.Optimizer,
    transitions: list[Any],
    gamma: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict[str, float]:
    rewards = [float(transition.reward) for transition in transitions]
    returns = _build_returns(rewards, gamma=gamma, device=device)

    log_probs = torch.stack([transition.log_prob for transition in transitions]).to(device)
    values = torch.stack([transition.value for transition in transitions]).to(device)
    entropies = torch.stack([transition.entropy for transition in transitions]).to(device)

    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_grad_norm,
    )
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropy_bonus.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()) if hasattr(grad_norm, "detach") else float(grad_norm),
    }


def _evaluate_policy(
    *,
    runner: AdaptiveCommitteeRunner,
    policy: TorchControllerPolicy,
    tasks: list[TaskSpec],
    limit: int,
) -> dict[str, float]:
    selected = tasks[: max(1, min(limit, len(tasks)))]
    rewards: list[float] = []
    qualities: list[float] = []
    tokens: list[float] = []
    elapsed: list[float] = []
    decisions: list[float] = []

    with torch.no_grad():
        for task in selected:
            rollout = runner.rollout_task(task, policy, deterministic=True, track_grad=False)
            rewards.append(float(rollout.total_reward))
            qualities.append(float(rollout.quality_score))
            tokens.append(float(rollout.trace["controller"]["used_tokens"]))
            elapsed.append(float(rollout.trace["total_elapsed_sec"]))
            decisions.append(float(rollout.trace["controller"]["decision_count"]))

    count = float(len(selected))
    return {
        "count": count,
        "mean_reward": sum(rewards) / count,
        "mean_quality": sum(qualities) / count,
        "mean_tokens": sum(tokens) / count,
        "mean_elapsed_sec": sum(elapsed) / count,
        "mean_decisions": sum(decisions) / count,
    }


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
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--budget-tokens", type=int, default=16000)
    parser.add_argument("--max-decisions", type=int, default=6)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--per-role-call-cap", type=int, default=2)
    parser.add_argument("--min-tokens-for-call", type=int, default=300)

    parser.add_argument("--token-step-penalty", type=float, default=0.02)
    parser.add_argument("--latency-step-penalty", type=float, default=0.01)
    parser.add_argument("--restart-penalty", type=float, default=0.03)
    parser.add_argument("--invalid-schema-penalty", type=float, default=0.02)
    parser.add_argument("--stop-without-finalize-penalty", type=float, default=0.08)
    parser.add_argument("--no-answer-penalty", type=float, default=0.20)
    parser.add_argument("--terminal-quality-weight", type=float, default=1.0)
    parser.add_argument("--terminal-token-penalty", type=float, default=0.20)
    parser.add_argument("--terminal-latency-penalty", type=float, default=0.02)

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
    )
    reward = RewardConfig(
        token_step_penalty=args.token_step_penalty,
        latency_step_penalty=args.latency_step_penalty,
        restart_penalty=args.restart_penalty,
        invalid_schema_penalty=args.invalid_schema_penalty,
        stop_without_finalize_penalty=args.stop_without_finalize_penalty,
        no_answer_penalty=args.no_answer_penalty,
        terminal_quality_weight=args.terminal_quality_weight,
        terminal_token_penalty=args.terminal_token_penalty,
        terminal_latency_penalty=args.terminal_latency_penalty,
    )
    runner = AdaptiveCommitteeRunner(config=config, client=client, runtime=runtime, reward=reward)

    model = TorchControllerModel(input_dim=runner.state_size, hidden_size=args.hidden_size)
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
        },
    )

    if args.warmstart_heuristic:
        warm_policy = HeuristicControllerPolicy()
        warm_rollout = runner.rollout_task(train_tasks[0], warm_policy, deterministic=True, track_grad=False)
        _write_json(outdir / "warmstart_heuristic_trace.json", warm_rollout.trace)
        print("wrote warmstart_heuristic_trace.json")

    if args.evaluate_only:
        if not args.resume:
            raise ValueError("--evaluate-only requires --resume <checkpoint_path>.")
        metrics = _evaluate_policy(
            runner=runner,
            policy=policy,
            tasks=eval_tasks,
            limit=min(args.eval_task_count, len(eval_tasks)),
        )
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
        rollout = runner.rollout_task(task, policy, deterministic=False, track_grad=True)
        train_metrics = _train_step(
            optimizer=optimizer,
            transitions=rollout.transitions,
            gamma=args.gamma,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )
        record = {
            "episode": episode,
            "task_id": task.task_id,
            "quality_score": float(rollout.quality_score),
            "total_reward": float(rollout.total_reward),
            "total_tokens": int(rollout.trace["controller"]["used_tokens"]),
            "decision_count": int(rollout.trace["controller"]["decision_count"]),
            "elapsed_sec": float(rollout.trace["total_elapsed_sec"]),
            **train_metrics,
        }
        _append_jsonl(train_log_path, record)
        print(
            f"episode={episode} reward={record['total_reward']:.4f} quality={record['quality_score']:.4f} "
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
            eval_metrics = _evaluate_policy(
                runner=runner,
                policy=policy,
                tasks=eval_tasks,
                limit=min(args.eval_task_count, len(eval_tasks)),
            )
            eval_metrics["episode"] = episode
            _append_jsonl(eval_log_path, eval_metrics)
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
