from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import random
import time
from typing import Any

from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig
from committee_llm.evaluation import evaluate_answer
from committee_llm.orchestrator import (
    _behavior_metrics,
    _build_role_prompt,
    _build_trace_summary,
    _extract_section,
    _schema_validity,
)
from committee_llm.tasks import TaskSpec

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except Exception:  # pragma: no cover - optional dependency for non-training paths
    torch = None
    nn = None
    Categorical = None


ACTION_CALL_RESEARCHER = 0
ACTION_CALL_ANALYST = 1
ACTION_CALL_CRITIC = 2
ACTION_CALL_FINALIZE = 3
ACTION_RESTART_FROM_PLAN = 4
ACTION_STOP = 5

ACTION_NAMES = (
    "call_researcher",
    "call_analyst",
    "call_critic",
    "call_finalize",
    "restart_from_plan",
    "stop",
)


def action_name(action_index: int) -> str:
    if 0 <= action_index < len(ACTION_NAMES):
        return ACTION_NAMES[action_index]
    return f"unknown_{action_index}"


def _usage_total_tokens(usage: dict[str, Any] | None, text: str) -> int:
    if isinstance(usage, dict):
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
    # Very rough fallback when backend omits usage
    return max(1, int(len(text.split()) * 1.5))


def _quality_score(evaluation: dict[str, Any]) -> float:
    reference = evaluation.get("reference", {})
    if not isinstance(reference, dict):
        return 0.0
    metrics = reference.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0

    choice_accuracy = metrics.get("choice_accuracy")
    if isinstance(choice_accuracy, (int, float)) and not isinstance(choice_accuracy, bool):
        return float(choice_accuracy)

    answer_em = metrics.get("answer_em")
    answer_f1 = metrics.get("answer_f1")
    em = float(answer_em) if isinstance(answer_em, (int, float)) and not isinstance(answer_em, bool) else 0.0
    f1 = float(answer_f1) if isinstance(answer_f1, (int, float)) and not isinstance(answer_f1, bool) else 0.0
    return 0.5 * em + 0.5 * f1


@dataclass(slots=True)
class AdaptiveRuntimeConfig:
    budget_tokens: int = 16000
    max_decisions: int = 6
    max_restarts: int = 1
    per_role_call_cap: int = 2
    min_tokens_for_call: int = 300
    max_backend_retries: int = 2


@dataclass(slots=True)
class RewardConfig:
    token_step_penalty: float = 0.02
    latency_step_penalty: float = 0.01
    restart_penalty: float = 0.03
    invalid_schema_penalty: float = 0.02
    stop_without_finalize_penalty: float = 0.08
    no_answer_penalty: float = 0.20
    terminal_quality_weight: float = 1.0
    terminal_token_penalty: float = 0.20
    terminal_latency_penalty: float = 0.02


@dataclass(slots=True)
class PolicyDecision:
    action: int
    log_prob: Any | None = None
    value: Any | None = None
    entropy: Any | None = None
    probs: list[float] | None = None


@dataclass(slots=True)
class RolloutTransition:
    log_prob: Any | None
    value: Any | None
    entropy: Any | None
    reward: float


@dataclass(slots=True)
class AdaptiveRollout:
    trace: dict[str, Any]
    transitions: list[RolloutTransition]
    total_reward: float
    quality_score: float


class ControllerPolicy:
    name = "controller_policy"

    def select_action(
        self,
        state_vector: list[float],
        action_mask: list[bool],
        *,
        deterministic: bool,
        track_grad: bool,
    ) -> PolicyDecision:
        raise NotImplementedError


class RandomControllerPolicy(ControllerPolicy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_action(
        self,
        state_vector: list[float],
        action_mask: list[bool],
        *,
        deterministic: bool,
        track_grad: bool,
    ) -> PolicyDecision:
        del state_vector, deterministic, track_grad
        allowed = [index for index, allowed_flag in enumerate(action_mask) if allowed_flag]
        if not allowed:
            return PolicyDecision(action=ACTION_STOP)
        # Mildly prefer non-stop actions unless stop is the only available option.
        non_stop = [index for index in allowed if index != ACTION_STOP]
        action = self._rng.choice(non_stop if non_stop else allowed)
        return PolicyDecision(action=action)


class HeuristicControllerPolicy(ControllerPolicy):
    name = "heuristic"

    def select_action(
        self,
        state_vector: list[float],
        action_mask: list[bool],
        *,
        deterministic: bool,
        track_grad: bool,
    ) -> PolicyDecision:
        del deterministic, track_grad
        # State layout is defined in AdaptiveCommitteeRunner._state_vector.
        remaining_budget = state_vector[1]
        has_research = state_vector[4] > 0.0
        has_analyst = state_vector[5] > 0.0
        has_critic = state_vector[6] > 0.0
        decisions_used = state_vector[0]

        preferred = []
        if not has_research:
            preferred = [ACTION_CALL_RESEARCHER]
        elif not has_analyst:
            preferred = [ACTION_CALL_ANALYST]
        elif not has_critic and remaining_budget > 0.25:
            preferred = [ACTION_CALL_CRITIC]
        elif remaining_budget < 0.18 or decisions_used > 0.75:
            preferred = [ACTION_CALL_FINALIZE, ACTION_STOP]
        else:
            preferred = [ACTION_CALL_FINALIZE, ACTION_CALL_CRITIC, ACTION_STOP]

        for action in preferred:
            if 0 <= action < len(action_mask) and action_mask[action]:
                return PolicyDecision(action=action)

        for action, allowed in enumerate(action_mask):
            if allowed:
                return PolicyDecision(action=action)
        return PolicyDecision(action=ACTION_STOP)


if nn is not None:
    class TorchControllerModel(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int = 128) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_size, len(ACTION_NAMES))
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, state_tensor: Any) -> tuple[Any, Any]:
            hidden = self.layers(state_tensor)
            logits = self.policy_head(hidden)
            value = self.value_head(hidden).squeeze(-1)
            return logits, value


    class TorchControllerPolicy(ControllerPolicy):
        name = "learned_torch"

        def __init__(self, model: Any, device: str = "cpu") -> None:
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)

        def select_action(
            self,
            state_vector: list[float],
            action_mask: list[bool],
            *,
            deterministic: bool,
            track_grad: bool,
        ) -> PolicyDecision:
            if not any(action_mask):
                return PolicyDecision(action=ACTION_STOP)

            if track_grad:
                context = torch.enable_grad()
            else:
                context = torch.no_grad()

            with context:
                state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
                logits, values = self.model(state_tensor)
                masked_logits = logits.clone()
                for action_index, allowed in enumerate(action_mask):
                    if not allowed:
                        masked_logits[0, action_index] = -1e9
                distribution = Categorical(logits=masked_logits)

                if deterministic:
                    action_tensor = torch.argmax(masked_logits, dim=-1)
                else:
                    action_tensor = distribution.sample()

                log_prob = distribution.log_prob(action_tensor)
                entropy = distribution.entropy()
                probs = torch.softmax(masked_logits, dim=-1).detach().cpu()[0].tolist()

            return PolicyDecision(
                action=int(action_tensor.item()),
                log_prob=log_prob if track_grad else log_prob.detach(),
                value=values if track_grad else values.detach(),
                entropy=entropy if track_grad else entropy.detach(),
                probs=[float(item) for item in probs],
            )
else:
    class TorchControllerModel:  # pragma: no cover - executed only when torch missing
        def __init__(self, input_dim: int, hidden_size: int = 128) -> None:
            del input_dim, hidden_size
            raise RuntimeError("Torch is required for TorchControllerModel.")


    class TorchControllerPolicy(ControllerPolicy):  # pragma: no cover - executed only when torch missing
        def __init__(self, model: Any, device: str = "cpu") -> None:
            del model, device
            raise RuntimeError("Torch is required for TorchControllerPolicy.")

        def select_action(
            self,
            state_vector: list[float],
            action_mask: list[bool],
            *,
            deterministic: bool,
            track_grad: bool,
        ) -> PolicyDecision:
            del state_vector, action_mask, deterministic, track_grad
            raise RuntimeError("Torch is required for TorchControllerPolicy.")


class AdaptiveCommitteeRunner:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        client: OpenAICompatibleChatClient,
        runtime: AdaptiveRuntimeConfig | None = None,
        reward: RewardConfig | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.runtime = runtime or AdaptiveRuntimeConfig()
        self.reward = reward or RewardConfig()

    @property
    def state_size(self) -> int:
        return 28

    def _state_vector(
        self,
        *,
        task: TaskSpec,
        decision_index: int,
        remaining_budget: int,
        used_tokens: int,
        call_counts: dict[str, int],
        schema_validity: dict[str, bool],
        last_action: int | None,
    ) -> list[float]:
        max_decisions = max(1, self.runtime.max_decisions)
        budget = max(1, self.runtime.budget_tokens)
        cap = max(1, self.runtime.per_role_call_cap)

        task_type_mcq = 1.0 if task.task_type == "multiple_choice" else 0.0
        task_type_open = 1.0 if task.task_type == "open_qa" else 0.0
        benchmark_name = (task.benchmark_name or "").lower()
        bench_hotpot = 1.0 if "hotpot" in benchmark_name else 0.0
        bench_musique = 1.0 if "musique" in benchmark_name else 0.0
        bench_gpqa = 1.0 if "gpqa" in benchmark_name else 0.0

        vector = [
            float(decision_index) / float(max_decisions),  # 0
            max(0.0, min(1.0, float(remaining_budget) / float(budget))),  # 1
            max(0.0, min(2.0, float(used_tokens) / float(budget))),  # 2
            1.0 if "coordinator_plan" in schema_validity else 0.0,  # 3
            1.0 if call_counts["researcher"] > 0 else 0.0,  # 4
            1.0 if call_counts["analyst"] > 0 else 0.0,  # 5
            1.0 if call_counts["critic"] > 0 else 0.0,  # 6
            1.0 if call_counts["coordinator_finalize"] > 0 else 0.0,  # 7
            float(call_counts["researcher"]) / float(cap),  # 8
            float(call_counts["analyst"]) / float(cap),  # 9
            float(call_counts["critic"]) / float(cap),  # 10
            float(call_counts["coordinator_plan"]) / float(max(1, self.runtime.max_restarts + 1)),  # 11
            1.0 if schema_validity.get("coordinator_plan") else 0.0,  # 12
            1.0 if schema_validity.get("researcher") else 0.0,  # 13
            1.0 if schema_validity.get("analyst") else 0.0,  # 14
            1.0 if schema_validity.get("critic") else 0.0,  # 15
            1.0 if schema_validity.get("coordinator_finalize") else 0.0,  # 16
            task_type_open,  # 17
            task_type_mcq,  # 18
            bench_hotpot,  # 19
            bench_musique,  # 20
            bench_gpqa,  # 21
        ]
        for action_index in range(len(ACTION_NAMES)):
            vector.append(1.0 if last_action == action_index else 0.0)  # 22-27
        return vector

    def _action_mask(
        self,
        *,
        remaining_budget: int,
        decision_index: int,
        restart_count: int,
        call_counts: dict[str, int],
    ) -> list[bool]:
        mask = [True] * len(ACTION_NAMES)

        has_research = call_counts["researcher"] > 0
        has_analyst = call_counts["analyst"] > 0
        has_final = call_counts["coordinator_finalize"] > 0

        if has_final:
            return [False, False, False, False, False, True]

        if decision_index >= self.runtime.max_decisions:
            return [False, False, False, False, False, True]

        if remaining_budget < self.runtime.min_tokens_for_call:
            return [False, False, False, False, False, True]

        if call_counts["researcher"] >= self.runtime.per_role_call_cap:
            mask[ACTION_CALL_RESEARCHER] = False
        if call_counts["analyst"] >= self.runtime.per_role_call_cap:
            mask[ACTION_CALL_ANALYST] = False
        if call_counts["critic"] >= self.runtime.per_role_call_cap:
            mask[ACTION_CALL_CRITIC] = False
        if call_counts["coordinator_finalize"] >= 1:
            mask[ACTION_CALL_FINALIZE] = False

        if not has_research:
            mask[ACTION_CALL_ANALYST] = False
        if not has_analyst:
            mask[ACTION_CALL_CRITIC] = False
            mask[ACTION_CALL_FINALIZE] = False

        if restart_count >= self.runtime.max_restarts:
            mask[ACTION_RESTART_FROM_PLAN] = False

        if not any(mask):
            return [False, False, False, False, False, True]
        return mask

    def _call_role(
        self,
        *,
        role_name: str,
        task: TaskSpec,
        stage_outputs: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        role_settings = self.config.resolve_role_settings(role_name)
        role_config = self.config.roles[role_name]
        system_prompt = self.config.load_system_prompt(role_name)
        user_prompt = _build_role_prompt(role_name, task, stage_outputs, self.config)
        max_tokens = int(role_settings["max_tokens"])
        error_messages: list[str] = []

        last_error: Exception | None = None
        for attempt in range(self.runtime.max_backend_retries + 1):
            try:
                result = self.client.chat(
                    model=str(role_settings["model"]),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(role_settings["temperature"]),
                    max_tokens=max_tokens,
                    extra_body=role_settings.get("extra_body"),
                )
                schema_validation = _schema_validity(result.content, self.config.get_role_schema(role_name))
                step_payload = {
                    "role": role_name,
                    "role_specialization": role_config.role_specialization,
                    "checkpoint_name": role_config.checkpoint_name,
                    "model": result.model,
                    "temperature": role_settings["temperature"],
                    "max_tokens": max_tokens,
                    "extra_body": role_settings.get("extra_body"),
                    "system_prompt": system_prompt,
                    "local_context_view": list(self.config.get_local_context_view(role_name)),
                    "output_schema": list(self.config.get_role_schema(role_name)),
                    "schema_validation": schema_validation,
                    "user_prompt": user_prompt,
                    "response": result.content,
                    "elapsed_sec": result.elapsed_sec,
                    "usage": result.usage,
                    "attempt": attempt,
                }
                return step_payload, schema_validation
            except Exception as exc:
                last_error = exc
                text = str(exc)
                error_messages.append(text)
                if "max_tokens" in text and "too large" in text and max_tokens > 128:
                    max_tokens = max(128, int(max_tokens * 0.6))
                    continue
                break

        joined = " | ".join(error_messages[-3:])
        raise RuntimeError(f"Role '{role_name}' failed after retries: {joined}") from last_error

    def _derive_final_answer(self, stage_outputs: dict[str, str]) -> tuple[str, str]:
        if "coordinator_finalize" in stage_outputs:
            final_stage = stage_outputs["coordinator_finalize"]
            final_answer = _extract_section(final_stage, "Final Answer") or final_stage.strip()
            return final_stage, final_answer
        if "analyst" in stage_outputs:
            analyst_text = stage_outputs["analyst"]
            final_answer = (
                _extract_section(analyst_text, "Draft Deliverable")
                or _extract_section(analyst_text, "Core Answer")
                or analyst_text.strip()
            )
            return analyst_text, final_answer
        if "researcher" in stage_outputs:
            researcher_text = stage_outputs["researcher"]
            final_answer = _extract_section(researcher_text, "Research Handoff") or researcher_text.strip()
            return researcher_text, final_answer
        fallback = stage_outputs.get("coordinator_plan", "")
        return fallback, fallback.strip()

    def rollout_task(
        self,
        task: TaskSpec,
        policy: ControllerPolicy,
        *,
        deterministic: bool,
        track_grad: bool,
    ) -> AdaptiveRollout:
        run_id = task.task_id or f"adaptive-{int(time.time() * 1000)}"
        started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()

        stage_outputs: dict[str, str] = {}
        stage_schemas: dict[str, dict[str, Any]] = {}
        schema_valid_flags: dict[str, bool] = {}
        steps: list[dict[str, Any]] = []
        transitions: list[RolloutTransition] = []
        controller_events: list[dict[str, Any]] = []

        call_counts = {
            "coordinator_plan": 0,
            "researcher": 0,
            "analyst": 0,
            "critic": 0,
            "coordinator_finalize": 0,
        }

        used_tokens = 0
        restart_count = 0
        decision_index = 0
        last_action: int | None = None

        # Initial mandatory plan call for a stable starting state.
        plan_step, plan_schema = self._call_role(role_name="coordinator_plan", task=task, stage_outputs=stage_outputs)
        stage_outputs["coordinator_plan"] = plan_step["response"]
        stage_schemas["coordinator_plan"] = plan_schema
        schema_valid_flags["coordinator_plan"] = bool(plan_schema["is_valid"])
        steps.append(plan_step)
        call_counts["coordinator_plan"] += 1
        used_tokens += _usage_total_tokens(plan_step.get("usage"), plan_step.get("response", ""))
        remaining_budget = self.runtime.budget_tokens - used_tokens

        done = False
        while not done and decision_index < self.runtime.max_decisions:
            action_mask = self._action_mask(
                remaining_budget=remaining_budget,
                decision_index=decision_index,
                restart_count=restart_count,
                call_counts=call_counts,
            )
            state_vector = self._state_vector(
                task=task,
                decision_index=decision_index,
                remaining_budget=remaining_budget,
                used_tokens=used_tokens,
                call_counts=call_counts,
                schema_validity=schema_valid_flags,
                last_action=last_action,
            )

            decision = policy.select_action(
                state_vector=state_vector,
                action_mask=action_mask,
                deterministic=deterministic,
                track_grad=track_grad,
            )
            selected_action = decision.action
            invalid_action = not (0 <= selected_action < len(action_mask)) or not action_mask[selected_action]
            if invalid_action:
                selected_action = ACTION_STOP

            step_reward = 0.0
            tokens_spent = 0
            latency_spent = 0.0
            role_called = None
            schema_valid = None

            if selected_action == ACTION_STOP:
                done = True
                if call_counts["coordinator_finalize"] == 0:
                    step_reward -= self.reward.stop_without_finalize_penalty
            elif selected_action == ACTION_RESTART_FROM_PLAN:
                restart_count += 1
                role_called = "coordinator_plan"
                restart_step, restart_schema = self._call_role(
                    role_name="coordinator_plan", task=task, stage_outputs=stage_outputs
                )
                stage_outputs["coordinator_plan"] = restart_step["response"]
                # Clear downstream outputs when plan restarts.
                for key in ("researcher", "analyst", "critic", "coordinator_finalize"):
                    stage_outputs.pop(key, None)
                    stage_schemas.pop(key, None)
                    schema_valid_flags.pop(key, None)
                    call_counts[key] = 0
                stage_schemas["coordinator_plan"] = restart_schema
                schema_valid_flags["coordinator_plan"] = bool(restart_schema["is_valid"])
                steps.append(restart_step)
                call_counts["coordinator_plan"] += 1
                tokens_spent = _usage_total_tokens(restart_step.get("usage"), restart_step.get("response", ""))
                latency_spent = float(restart_step.get("elapsed_sec", 0.0))
                schema_valid = bool(restart_schema["is_valid"])
                step_reward -= self.reward.restart_penalty
            else:
                role_map = {
                    ACTION_CALL_RESEARCHER: "researcher",
                    ACTION_CALL_ANALYST: "analyst",
                    ACTION_CALL_CRITIC: "critic",
                    ACTION_CALL_FINALIZE: "coordinator_finalize",
                }
                role_called = role_map[selected_action]
                role_step, role_schema = self._call_role(
                    role_name=role_called,
                    task=task,
                    stage_outputs=stage_outputs,
                )
                stage_outputs[role_called] = role_step["response"]
                stage_schemas[role_called] = role_schema
                schema_valid_flags[role_called] = bool(role_schema["is_valid"])
                steps.append(role_step)
                call_counts[role_called] += 1
                tokens_spent = _usage_total_tokens(role_step.get("usage"), role_step.get("response", ""))
                latency_spent = float(role_step.get("elapsed_sec", 0.0))
                schema_valid = bool(role_schema["is_valid"])
                if selected_action == ACTION_CALL_FINALIZE:
                    done = True

            if tokens_spent > 0:
                step_reward -= self.reward.token_step_penalty * (
                    float(tokens_spent) / float(max(1, self.runtime.budget_tokens))
                )
            if latency_spent > 0:
                step_reward -= self.reward.latency_step_penalty * (latency_spent / 60.0)
            if schema_valid is False:
                step_reward -= self.reward.invalid_schema_penalty
            if invalid_action:
                step_reward -= self.reward.invalid_schema_penalty

            used_tokens += tokens_spent
            remaining_budget = self.runtime.budget_tokens - used_tokens
            if remaining_budget <= 0:
                done = True

            transitions.append(
                RolloutTransition(
                    log_prob=decision.log_prob,
                    value=decision.value,
                    entropy=decision.entropy,
                    reward=step_reward,
                )
            )
            controller_events.append(
                {
                    "decision_index": decision_index,
                    "state_vector": state_vector,
                    "action_mask": action_mask,
                    "selected_action": selected_action,
                    "selected_action_name": action_name(selected_action),
                    "invalid_action": invalid_action,
                    "role_called": role_called,
                    "schema_valid": schema_valid,
                    "tokens_spent": tokens_spent,
                    "latency_spent_sec": latency_spent,
                    "remaining_budget_tokens": max(0, remaining_budget),
                    "log_prob": float(decision.log_prob.detach().cpu().item())
                    if hasattr(decision.log_prob, "detach")
                    else (float(decision.log_prob) if decision.log_prob is not None else None),
                    "value": float(decision.value.detach().cpu().item())
                    if hasattr(decision.value, "detach")
                    else (float(decision.value) if decision.value is not None else None),
                    "entropy": float(decision.entropy.detach().cpu().item())
                    if hasattr(decision.entropy, "detach")
                    else (float(decision.entropy) if decision.entropy is not None else None),
                    "probs": decision.probs,
                }
            )

            last_action = selected_action
            decision_index += 1

        final_stage, final_answer = self._derive_final_answer(stage_outputs)
        evaluation = evaluate_answer(task, final_answer)
        quality = _quality_score(evaluation)
        elapsed_sec = time.perf_counter() - run_start

        terminal_reward = self.reward.terminal_quality_weight * quality
        terminal_reward -= self.reward.terminal_token_penalty * (
            float(used_tokens) / float(max(1, self.runtime.budget_tokens))
        )
        terminal_reward -= self.reward.terminal_latency_penalty * (elapsed_sec / 60.0)
        if not final_answer.strip():
            terminal_reward -= self.reward.no_answer_penalty
        if transitions:
            transitions[-1].reward += terminal_reward

        summary = _build_trace_summary(steps)
        summary["behavior_metrics"] = _behavior_metrics(task, stage_outputs, stage_schemas)
        summary["schema_validity"] = {
            role_name: {
                "is_valid": payload["is_valid"],
                "missing_headers": payload["missing_headers"],
            }
            for role_name, payload in stage_schemas.items()
        }

        trace = {
            "run_id": run_id,
            "architecture": "architecture_2_adaptive_controller",
            "architecture_spec": {
                "controller_policy": policy.name,
                "action_space": list(ACTION_NAMES),
                "runtime": {
                    "budget_tokens": self.runtime.budget_tokens,
                    "max_decisions": self.runtime.max_decisions,
                    "max_restarts": self.runtime.max_restarts,
                    "per_role_call_cap": self.runtime.per_role_call_cap,
                    "min_tokens_for_call": self.runtime.min_tokens_for_call,
                },
                "reward": {
                    "token_step_penalty": self.reward.token_step_penalty,
                    "latency_step_penalty": self.reward.latency_step_penalty,
                    "restart_penalty": self.reward.restart_penalty,
                    "invalid_schema_penalty": self.reward.invalid_schema_penalty,
                    "stop_without_finalize_penalty": self.reward.stop_without_finalize_penalty,
                    "no_answer_penalty": self.reward.no_answer_penalty,
                    "terminal_quality_weight": self.reward.terminal_quality_weight,
                    "terminal_token_penalty": self.reward.terminal_token_penalty,
                    "terminal_latency_penalty": self.reward.terminal_latency_penalty,
                },
            },
            "config_name": self.config.name,
            "config_path": str(self.config.source_path),
            "started_at_utc": started_at,
            "total_elapsed_sec": elapsed_sec,
            "input": task.prompt,
            "task": task.to_trace_payload(),
            "metadata": dict(task.metadata),
            "steps": steps,
            "controller": {
                "decisions": controller_events,
                "decision_count": len(controller_events),
                "restart_count": restart_count,
                "used_tokens": used_tokens,
                "remaining_budget_tokens": max(0, remaining_budget),
            },
            "final_stage": final_stage,
            "final_answer": final_answer,
            "evaluation": evaluation,
            "summary": summary,
        }

        total_reward = sum(transition.reward for transition in transitions)
        return AdaptiveRollout(
            trace=trace,
            transitions=transitions,
            total_reward=float(total_reward),
            quality_score=float(quality),
        )
