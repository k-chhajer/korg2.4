from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import random
import re
import string
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


REASONING_PHASES = [
    "decomposing",
    "hypothesizing",
    "conflicting",
    "converging",
    "verifying",
    "stuck",
]

PHASE_TO_IDX = {phase: index for index, phase in enumerate(REASONING_PHASES)}
N_PHASES = len(REASONING_PHASES)
SEMANTIC_EMBED_DIM = 384
TASK_PROGRESS_DIM = 2


def action_name(action_index: int) -> str:
    if 0 <= action_index < len(ACTION_NAMES):
        return ACTION_NAMES[action_index]
    return f"unknown_{action_index}"


def _normalize_answer(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = " ".join(normalized.split())
    return normalized


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = sum(pred_tokens.count(token) for token in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(token) for token in common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class SemanticStateEncoder:
    """
    Frozen local encoder for question-conditioned recent debate semantics.

    The controller trains only a small projection over this vector. If the
    optional dependency or model cache is unavailable, this safely returns
    zeros so training can still run with the structural/phase state.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = SEMANTIC_EMBED_DIM) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model: Any | None = None
        self._disabled = False
        self.last_error: str | None = None

    def _zeros(self) -> list[float]:
        return [0.0] * self.embedding_dim

    def _load_model(self) -> Any | None:
        if self._disabled:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except TypeError:
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:
            self._disabled = True
            self.last_error = str(exc)
            return None

    def encode(self, *, question: str, transcript: list[dict[str, str]]) -> list[float]:
        recent = transcript[-3:] if len(transcript) >= 3 else transcript
        recent_text = " | ".join(
            f"{turn.get('role', 'unknown')}: {str(turn.get('content', ''))[:300]}" for turn in recent
        )
        text = f"Question: {question[:400]} | Recent debate: {recent_text}"
        return self.encode_text(text)

    def encode_text(self, text: str) -> list[float]:
        model = self._load_model()
        if model is None:
            return self._zeros()
        try:
            embedding = model.encode(text, show_progress_bar=False)
            values = [float(item) for item in embedding.tolist()]
        except Exception as exc:
            self.last_error = str(exc)
            return self._zeros()

        if len(values) < self.embedding_dim:
            values.extend([0.0] * (self.embedding_dim - len(values)))
        return values[: self.embedding_dim]

    def encode_delta(self, transcript: list[dict[str, str]]) -> list[float]:
        if len(transcript) < 2:
            return self._zeros()
        previous_text = f"{transcript[-2].get('role', 'unknown')}: {str(transcript[-2].get('content', ''))[:300]}"
        current_text = f"{transcript[-1].get('role', 'unknown')}: {str(transcript[-1].get('content', ''))[:300]}"
        previous = self.encode_text(previous_text)
        current = self.encode_text(current_text)
        return [current_value - previous_value for current_value, previous_value in zip(current, previous)]


def compute_task_progress_features(question: str, transcript: list[dict[str, str]]) -> list[float]:
    """
    Cheap semantic progress proxy for multi-hop QA.

    Feature 0: fraction of capitalized question entities observed in the transcript.
    Feature 1: whether a numeric/date-like answer candidate has appeared.
    """
    entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    stop_entities = {
        "A",
        "An",
        "And",
        "Are",
        "Did",
        "Do",
        "Does",
        "For",
        "How",
        "In",
        "Is",
        "Of",
        "The",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Whom",
        "Whose",
        "Why",
    }
    question_entities = {
        entity.strip()
        for entity in re.findall(entity_pattern, question)
        if entity.strip() and entity.strip() not in stop_entities
    }
    transcript_text = " ".join(str(turn.get("content", "")) for turn in transcript)
    covered = sum(1 for entity in question_entities if entity in transcript_text)
    coverage = covered / max(len(question_entities), 1)
    has_numeric_answer = bool(re.search(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b", transcript_text))
    return [float(max(0.0, min(1.0, coverage))), float(has_numeric_answer)]


def compute_verifiable_reward(
    predicted_answer: str,
    gold_answer: str | list[str],
    task_type: str,
) -> float:
    """
    RLVR reward: binary/F1 from ground truth only.
    No shaped components. No LLM judge.
    """
    pred = _normalize_answer(predicted_answer)
    normalized_task_type = task_type.strip().lower()

    if normalized_task_type in {"gpqa", "arc"}:
        gold_raw = gold_answer if isinstance(gold_answer, str) else (gold_answer[0] if gold_answer else "")
        gold = _normalize_answer(gold_raw)
        pred_letter = re.search(r"\b([a-d])\b", pred)
        gold_letter = re.search(r"\b([a-d])\b", gold)
        if pred_letter and gold_letter:
            return 1.0 if pred_letter.group(1).upper() == gold_letter.group(1).upper() else 0.0
        return 1.0 if pred == gold else 0.0

    gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    candidates = [candidate for candidate in gold_list if isinstance(candidate, str)]
    if not candidates:
        return 0.0
    best_f1 = max(_token_f1(pred, _normalize_answer(candidate)) for candidate in candidates)
    return best_f1 if best_f1 > 0.6 else best_f1 * 0.5


def classify_reasoning_phase(
    transcript: list[dict[str, str]],
    client: OpenAICompatibleChatClient,
    cheap_model: str,
) -> int:
    """Returns phase index. Falls back to 'hypothesizing' (index 1) on failure."""
    recent = transcript[-3:] if len(transcript) >= 3 else transcript
    transcript_text = "\n".join(
        f"{turn.get('role', 'unknown').upper()}: {str(turn.get('content', ''))[:200]}" for turn in recent
    )

    prompt = f"""You are analyzing a multi-agent reasoning debate.

Current debate transcript (recent turns):
{transcript_text}

Classify the current reasoning phase as exactly one of:
- decomposing: agents are breaking down the question
- hypothesizing: candidate answers are being proposed
- conflicting: agents disagree, competing answers exist
- converging: agreement is emerging
- verifying: checking/stress-testing the leading answer
- stuck: circular or repetitive, no new information

Respond with ONLY the phase word, nothing else."""

    try:
        result = client.chat(
            model=cheap_model,
            messages=[
                {"role": "system", "content": "You classify reasoning phases."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        phase_word = result.content.strip().lower().split()[0] if result.content.strip() else "hypothesizing"
        phase_word = re.sub(r"[^a-z]", "", phase_word)
        return PHASE_TO_IDX.get(phase_word, 1)
    except Exception:
        return 1


def _usage_total_tokens(usage: dict[str, Any] | None, text: str) -> int:
    if isinstance(usage, dict):
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
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


def _infer_reward_task_type(task: TaskSpec) -> str:
    benchmark_name = (task.benchmark_name or "").strip().lower()
    if "gpqa" in benchmark_name:
        return "gpqa"
    if "arc" in benchmark_name:
        return "arc"
    if "musique" in benchmark_name:
        return "musique"
    if "hotpot" in benchmark_name:
        return "hotpotqa"
    if task.task_type == "multiple_choice":
        return "gpqa"
    return "hotpotqa"


def _extract_gold_answer(task: TaskSpec, task_type: str) -> str | list[str]:
    if task_type in {"gpqa", "arc"}:
        if task.correct_choice:
            return task.correct_choice
        if task.reference_answers:
            return task.reference_answers[0]
        return ""

    if task.reference_answers:
        return list(task.reference_answers)
    return ""


@dataclass(slots=True)
class AdaptiveRuntimeConfig:
    budget_tokens: int = 16000
    max_decisions: int = 6
    max_restarts: int = 1
    per_role_call_cap: int = 2
    min_tokens_for_call: int = 300
    max_backend_retries: int = 2
    cheap_model: str = "anthropic/claude-haiku-4-5-20251001"
    semantic_model: str = "all-MiniLM-L6-v2"
    use_semantic_embeddings: bool = True


@dataclass(slots=True)
class RewardConfig:
    reward_type: str = "verifiable"
    hotpotqa_f1_threshold: float = 0.6
    gpqa_match: str = "letter_exact"


@dataclass(slots=True)
class PolicyDecision:
    action: int
    log_prob: Any | None = None
    value: Any | None = None
    entropy: Any | None = None
    probs: list[float] | None = None
    action_logits: Any | None = None
    phase_logits: Any | None = None
    phase_label: int | None = None


@dataclass(slots=True)
class RolloutTransition:
    log_prob: Any | None
    value: Any | None
    entropy: Any | None
    reward: float
    action_logits: Any | None = None
    phase_logits: Any | None = None
    phase_label: Any | None = None


@dataclass(slots=True)
class AdaptiveRollout:
    trace: dict[str, Any]
    transitions: list[RolloutTransition]
    total_reward: float
    quality_score: float


class ControllerPolicy:
    name = "controller_policy"

    def start_episode(self) -> None:
        return None

    def select_action(
        self,
        state_vector: list[float],
        action_mask: list[bool],
        *,
        deterministic: bool,
        track_grad: bool,
        phase_idx: int | None = None,
        semantic_vector: list[float] | None = None,
        semantic_delta: list[float] | None = None,
        task_progress: list[float] | None = None,
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
        phase_idx: int | None = None,
        semantic_vector: list[float] | None = None,
        semantic_delta: list[float] | None = None,
        task_progress: list[float] | None = None,
    ) -> PolicyDecision:
        del state_vector, deterministic, track_grad, phase_idx, semantic_vector, semantic_delta, task_progress
        allowed = [index for index, allowed_flag in enumerate(action_mask) if allowed_flag]
        if not allowed:
            return PolicyDecision(action=ACTION_STOP)
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
        phase_idx: int | None = None,
        semantic_vector: list[float] | None = None,
        semantic_delta: list[float] | None = None,
        task_progress: list[float] | None = None,
    ) -> PolicyDecision:
        del deterministic, track_grad, phase_idx, semantic_vector, semantic_delta, task_progress
        remaining_budget = state_vector[1]
        has_research = state_vector[4] > 0.0
        has_analyst = state_vector[5] > 0.0
        has_critic = state_vector[6] > 0.0
        decisions_used = state_vector[0]

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
            self._hidden_state: Any | None = None

        def start_episode(self) -> None:
            if hasattr(self.model, "initial_hidden"):
                hidden = self.model.initial_hidden(batch_size=1)
                if hasattr(hidden, "to"):
                    hidden = hidden.to(self.device)
                self._hidden_state = hidden
            else:
                self._hidden_state = None

        def select_action(
            self,
            state_vector: list[float],
            action_mask: list[bool],
            *,
            deterministic: bool,
            track_grad: bool,
            phase_idx: int | None = None,
            semantic_vector: list[float] | None = None,
            semantic_delta: list[float] | None = None,
            task_progress: list[float] | None = None,
        ) -> PolicyDecision:
            if not any(action_mask):
                return PolicyDecision(action=ACTION_STOP)

            context = torch.enable_grad() if track_grad else torch.no_grad()
            with context:
                state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
                phase_label = 1 if phase_idx is None else int(phase_idx)
                phase_label = max(0, min(N_PHASES - 1, phase_label))

                if hasattr(self.model, "initial_hidden"):
                    if self._hidden_state is None:
                        self.start_episode()
                    phase_onehot = torch.zeros((1, N_PHASES), dtype=torch.float32, device=self.device)
                    phase_onehot[0, phase_label] = 1.0
                    if hasattr(self.model, "semantic_dim"):
                        semantic_dim = int(getattr(self.model, "semantic_dim"))
                        delta_dim = int(getattr(self.model, "semantic_delta_dim", semantic_dim))
                        progress_dim = int(getattr(self.model, "task_progress_dim", 0))
                        semantic_values = list(semantic_vector or [])
                        if len(semantic_values) < semantic_dim:
                            semantic_values.extend([0.0] * (semantic_dim - len(semantic_values)))
                        semantic_tensor = torch.tensor(
                            [semantic_values[:semantic_dim]], dtype=torch.float32, device=self.device
                        )
                        delta_values = list(semantic_delta or [])
                        if len(delta_values) < delta_dim:
                            delta_values.extend([0.0] * (delta_dim - len(delta_values)))
                        delta_tensor = torch.tensor(
                            [delta_values[:delta_dim]], dtype=torch.float32, device=self.device
                        )
                        progress_values = list(task_progress or [])
                        if len(progress_values) < progress_dim:
                            progress_values.extend([0.0] * (progress_dim - len(progress_values)))
                        progress_tensor = torch.tensor(
                            [progress_values[:progress_dim]], dtype=torch.float32, device=self.device
                        )
                        action_logits, values, hidden_next, phase_logits = self.model(
                            state_tensor,
                            phase_onehot,
                            semantic_tensor,
                            progress_tensor,
                            self._hidden_state,
                            delta_tensor,
                        )
                    else:
                        action_logits, values, hidden_next, phase_logits = self.model(
                            state_tensor, phase_onehot, self._hidden_state
                        )
                    self._hidden_state = hidden_next if track_grad else hidden_next.detach()
                else:
                    action_logits, values = self.model(state_tensor)
                    phase_logits = None

                masked_logits = action_logits.clone()
                for action_index, allowed in enumerate(action_mask):
                    if not allowed:
                        masked_logits[0, action_index] = -1e9

                distribution = Categorical(logits=masked_logits)
                action_tensor = torch.argmax(masked_logits, dim=-1) if deterministic else distribution.sample()
                log_prob = distribution.log_prob(action_tensor)
                entropy = distribution.entropy()
                probs = torch.softmax(masked_logits, dim=-1).detach().cpu()[0].tolist()

            return PolicyDecision(
                action=int(action_tensor.item()),
                log_prob=log_prob if track_grad else log_prob.detach(),
                value=values if track_grad else values.detach(),
                entropy=entropy if track_grad else entropy.detach(),
                probs=[float(item) for item in probs],
                action_logits=masked_logits if track_grad else masked_logits.detach(),
                phase_logits=phase_logits if track_grad else (phase_logits.detach() if phase_logits is not None else None),
                phase_label=phase_label,
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
            phase_idx: int | None = None,
            semantic_vector: list[float] | None = None,
            semantic_delta: list[float] | None = None,
            task_progress: list[float] | None = None,
        ) -> PolicyDecision:
            del state_vector, action_mask, deterministic, track_grad, phase_idx, semantic_vector, semantic_delta
            del task_progress
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
        self.semantic_encoder = (
            SemanticStateEncoder(model_name=self.runtime.semantic_model)
            if self.runtime.use_semantic_embeddings
            else None
        )

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
            float(decision_index) / float(max_decisions),
            max(0.0, min(1.0, float(remaining_budget) / float(budget))),
            max(0.0, min(2.0, float(used_tokens) / float(budget))),
            1.0 if "coordinator_plan" in schema_validity else 0.0,
            1.0 if call_counts["researcher"] > 0 else 0.0,
            1.0 if call_counts["analyst"] > 0 else 0.0,
            1.0 if call_counts["critic"] > 0 else 0.0,
            1.0 if call_counts["coordinator_finalize"] > 0 else 0.0,
            float(call_counts["researcher"]) / float(cap),
            float(call_counts["analyst"]) / float(cap),
            float(call_counts["critic"]) / float(cap),
            float(call_counts["coordinator_plan"]) / float(max(1, self.runtime.max_restarts + 1)),
            1.0 if schema_validity.get("coordinator_plan") else 0.0,
            1.0 if schema_validity.get("researcher") else 0.0,
            1.0 if schema_validity.get("analyst") else 0.0,
            1.0 if schema_validity.get("critic") else 0.0,
            1.0 if schema_validity.get("coordinator_finalize") else 0.0,
            task_type_open,
            task_type_mcq,
            bench_hotpot,
            bench_musique,
            bench_gpqa,
        ]
        for action_index in range(len(ACTION_NAMES)):
            vector.append(1.0 if last_action == action_index else 0.0)
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

        if hasattr(policy, "start_episode"):
            policy.start_episode()

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

            transcript = [
                {"role": str(step.get("role", "")), "content": str(step.get("response", ""))}
                for step in steps
            ]
            phase_idx = classify_reasoning_phase(
                transcript=transcript,
                client=self.client,
                cheap_model=self.runtime.cheap_model,
            )
            semantic_vector = (
                self.semantic_encoder.encode(question=task.prompt, transcript=transcript)
                if self.semantic_encoder is not None
                else [0.0] * SEMANTIC_EMBED_DIM
            )
            semantic_delta = (
                self.semantic_encoder.encode_delta(transcript)
                if self.semantic_encoder is not None
                else [0.0] * SEMANTIC_EMBED_DIM
            )
            task_progress = compute_task_progress_features(task.prompt, transcript)

            decision = policy.select_action(
                state_vector=state_vector,
                action_mask=action_mask,
                deterministic=deterministic,
                track_grad=track_grad,
                phase_idx=phase_idx,
                semantic_vector=semantic_vector,
                semantic_delta=semantic_delta,
                task_progress=task_progress,
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
            elif selected_action == ACTION_RESTART_FROM_PLAN:
                restart_count += 1
                role_called = "coordinator_plan"
                restart_step, restart_schema = self._call_role(
                    role_name="coordinator_plan", task=task, stage_outputs=stage_outputs
                )
                stage_outputs["coordinator_plan"] = restart_step["response"]
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

            used_tokens += tokens_spent
            remaining_budget = self.runtime.budget_tokens - used_tokens
            if remaining_budget <= 0:
                done = True

            phase_label_tensor = None
            if decision.phase_label is not None and torch is not None:
                phase_label_tensor = torch.tensor([int(decision.phase_label)], dtype=torch.long)
                if track_grad and hasattr(decision.action_logits, "device"):
                    phase_label_tensor = phase_label_tensor.to(decision.action_logits.device)

            transitions.append(
                RolloutTransition(
                    log_prob=decision.log_prob,
                    value=decision.value,
                    entropy=decision.entropy,
                    reward=step_reward,
                    action_logits=decision.action_logits,
                    phase_logits=decision.phase_logits,
                    phase_label=phase_label_tensor,
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
                    "phase_idx": phase_idx,
                    "phase_label": REASONING_PHASES[phase_idx],
                    "semantic_embedding_dim": len(semantic_vector),
                    "semantic_embedding_available": any(abs(value) > 0.0 for value in semantic_vector),
                    "semantic_delta_dim": len(semantic_delta),
                    "semantic_delta_available": any(abs(value) > 0.0 for value in semantic_delta),
                    "task_progress_features": {
                        "question_entity_coverage": task_progress[0],
                        "has_numeric_candidate": bool(task_progress[1]),
                    },
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

        reward_task_type = _infer_reward_task_type(task)
        gold_answer = _extract_gold_answer(task, reward_task_type)
        verifiable_reward = compute_verifiable_reward(
            predicted_answer=final_answer,
            gold_answer=gold_answer,
            task_type=reward_task_type,
        )

        if transitions:
            transitions[-1].reward += verifiable_reward

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
                    "cheap_model": self.runtime.cheap_model,
                    "semantic_model": self.runtime.semantic_model,
                    "use_semantic_embeddings": self.runtime.use_semantic_embeddings,
                    "semantic_embedding_dim": SEMANTIC_EMBED_DIM,
                    "semantic_delta_dim": SEMANTIC_EMBED_DIM,
                    "task_progress_dim": TASK_PROGRESS_DIM,
                    "semantic_encoder_error": self.semantic_encoder.last_error
                    if self.semantic_encoder is not None
                    else None,
                },
                "reward": {
                    "type": self.reward.reward_type,
                    "hotpotqa_f1_threshold": self.reward.hotpotqa_f1_threshold,
                    "gpqa_match": self.reward.gpqa_match,
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
                "action_sequence": [event["selected_action_name"] for event in controller_events],
                "phase_sequence": [event["phase_label"] for event in controller_events],
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
            quality_score=float(verifiable_reward if verifiable_reward is not None else quality),
        )
