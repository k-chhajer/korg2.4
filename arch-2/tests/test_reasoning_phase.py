"""Tests for phase classifier and GRU controller."""

import torch

from committee_llm.adaptive import N_PHASES, PHASE_TO_IDX, compute_task_progress_features
from committee_llm.train_controller import ReasoningAwareController


def test_controller_forward_shape() -> None:
    model = ReasoningAwareController(state_dim=28, n_phases=6, n_actions=6, hidden_size=128)
    state = torch.zeros(1, 28)
    phase = torch.zeros(1, 6)
    phase[0, 0] = 1.0
    hidden = model.initial_hidden()
    action_logits, value, hidden_new, phase_logits = model(state, phase, hidden)
    assert action_logits.shape == (1, 6)
    assert hidden_new.shape == (1, 128)
    assert phase_logits.shape == (1, 6)
    assert value.shape == (1,)


def test_controller_forward_shape_with_semantic_state() -> None:
    model = ReasoningAwareController(state_dim=28, n_phases=6, n_actions=6, hidden_size=128)
    state = torch.zeros(1, 28)
    phase = torch.zeros(1, 6)
    phase[0, 0] = 1.0
    semantic = torch.zeros(1, 384)
    delta = torch.zeros(1, 384)
    progress = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    hidden = model.initial_hidden()
    action_logits, value, hidden_new, phase_logits = model(state, phase, semantic, progress, hidden, delta)
    assert action_logits.shape == (1, 6)
    assert hidden_new.shape == (1, 128)
    assert phase_logits.shape == (1, 6)
    assert value.shape == (1,)


def test_hidden_state_updates() -> None:
    model = ReasoningAwareController()
    hidden_start = model.initial_hidden()
    state = torch.randn(1, 28)
    phase = torch.zeros(1, 6)
    phase[0, 2] = 1.0
    _, _, hidden_next, _ = model(state, phase, hidden_start)
    assert not torch.allclose(hidden_start, hidden_next)


def test_phase_taxonomy_complete() -> None:
    assert N_PHASES == 6
    assert "stuck" in PHASE_TO_IDX
    assert "converging" in PHASE_TO_IDX


def test_task_progress_features() -> None:
    features = compute_task_progress_features(
        "What city was Marie Curie born in?",
        [{"role": "researcher", "content": "Marie Curie was born in Warsaw in 1867."}],
    )
    assert features == [1.0, 1.0]
