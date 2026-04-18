# Codex Instructions: Arch2 Upgrades
**Project:** grok_multiagent / arch-2  
**Scope:** Everything below is isolated to `arch-2/`. Do not touch `arch-1/`.

---

## Overview of What You Are Doing

You are making four categories of changes:

1. **Housekeeping** — Fix all Arch1 label bleed into Arch2 files
2. **Reward replacement** — Rip out the shaped reward, replace with RLVR (binary verifiable reward only)
3. **Reasoning state awareness** — Add a GRU hidden state + phase classifier to the controller so it understands *where in the reasoning process* the debate is, not just what actions have been taken
4. **GRPO training** — Replace the current actor-critic loss with Group Relative Policy Optimization

Each section below is a self-contained task. Do them in order — later tasks depend on earlier ones.

---

## Task 1: Housekeeping — Fix Arch1 Label Bleed

These are mechanical find-and-replace fixes. No logic changes.

### 1a. `arch-2/implementation/committee_llm/config.py`
- Find all error strings and validation messages that say "Architecture 1", "Arch1", or "arch-1"
- Replace with "Architecture 2", "Arch2", "arch-2" respectively
- Do not change any logic, only string literals

### 1b. `arch-2/implementation/committee_llm/benchmark.py` line ~245
- The argparse description currently says something like `"Run focused Architecture 1 benchmarks…"`
- Change it to `"Run focused Architecture 2 benchmarks — frozen specialist committee with learned controller"`

### 1c. `arch-2/implementation/committee_llm/orchestrator.py` line ~323
- The committee runner trace prints an architecture string that says Arch1
- Change the string to say `"Architecture 2: frozen specialist committee + learned controller"`

### 1d. Global search
- Run `grep -r "arch-1\|arch_1\|Arch1\|Architecture 1" arch-2/` 
- Fix any remaining hits that are not intentional cross-references
- Leave any comment that says "unlike arch-1, this does X" — those are intentional

---

## Task 2: Replace Shaped Reward with RLVR

**File:** `arch-2/implementation/committee_llm/adaptive.py`

### 2a. Remove the existing reward components
The current reward has multiple shaped terms: step penalties for tokens, latency, restart, schema validity, early stop, plus terminal quality-token-latency terms. **Remove all of these.**

### 2b. Add a verifiable reward function

Add this function to `adaptive.py`:

```python
import re

def compute_verifiable_reward(
    predicted_answer: str,
    gold_answer: str | list[str],
    task_type: str,  # "hotpotqa" | "gpqa" | "musique" | "arc"
) -> float:
    """
    RLVR reward: binary/F1 from ground truth only.
    No shaped components. No LLM judge.
    """
    pred = _normalize_answer(predicted_answer)

    if task_type == "gpqa" or task_type == "arc":
        # Multiple choice — binary exact match on letter (A/B/C/D)
        gold = _normalize_answer(gold_answer if isinstance(gold_answer, str) else gold_answer[0])
        # Extract just the letter if model outputs "A) ..." or "(A)"
        pred_letter = re.search(r'\b([A-D])\b', pred)
        gold_letter = re.search(r'\b([A-D])\b', gold)
        if pred_letter and gold_letter:
            return 1.0 if pred_letter.group(1) == gold_letter.group(1) else 0.0
        return 1.0 if pred == gold else 0.0

    else:
        # HotpotQA / MuSiQue — token-level F1 (standard HotpotQA metric)
        gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]
        best_f1 = max(_token_f1(pred, _normalize_answer(g)) for g in gold_list)
        # Threshold: reward 1.0 if F1 > 0.6, else scaled F1
        # This avoids rewarding near-misses as strongly as exact matches
        return best_f1 if best_f1 > 0.6 else best_f1 * 0.5


def _normalize_answer(s: str) -> str:
    """Lower, strip punctuation, collapse whitespace."""
    import string
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 2c. Wire the verifiable reward into the episode loop

In the episode/rollout logic in `adaptive.py`, find where the reward is currently computed and replace it:

```python
# OLD — remove everything like this:
# reward = terminal_quality * w1 - token_penalty * w2 - latency_penalty * w3 ...

# NEW — only this:
reward = compute_verifiable_reward(
    predicted_answer=episode_final_answer,
    gold_answer=task["gold_answer"],      # field name may vary, check tasks.py
    task_type=task["benchmark"],          # "hotpotqa", "gpqa", etc.
)
```

### 2d. Remove reward hyperparameters from config

In `arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json`, remove any fields related to reward shaping weights (things like `token_penalty_weight`, `latency_weight`, `quality_weight`, etc.). The verifiable reward has no hyperparameters.

Also remove them from any config dataclass in `config.py` if they exist there.

---

## Task 3: Add Reasoning State Awareness to the Controller

**Files:** `arch-2/implementation/committee_llm/adaptive.py` (primary), `arch-2/implementation/committee_llm/train_controller.py`

This is the most significant architectural change. You are adding:
- A **phase classifier** (cheap LLM call) that labels where the debate currently is
- A **GRU cell** in the controller that maintains a learned hidden reasoning state
- An **auxiliary loss** that regularizes the GRU using phase labels

### 3a. Define the reasoning phase taxonomy

Add this to `adaptive.py`:

```python
REASONING_PHASES = [
    "decomposing",    # agents still breaking down the question into sub-problems
    "hypothesizing",  # candidate answers are being proposed
    "conflicting",    # agents are disagreeing, competing hypotheses exist
    "converging",     # agreement is emerging around one answer
    "verifying",      # checking / stress-testing the candidate answer
    "stuck",          # circular or repetitive outputs, no new information
]

PHASE_TO_IDX = {p: i for i, p in enumerate(REASONING_PHASES)}
N_PHASES = len(REASONING_PHASES)  # 6
```

### 3b. Add the phase classifier function

Add this to `adaptive.py`. It makes a cheap LLM call (use the smallest/cheapest model in your config — Haiku or Qwen 0.5B):

```python
async def classify_reasoning_phase(
    transcript: list[dict],   # list of {"role": agent_name, "content": "..."}
    client,                   # your existing LLM client
    cheap_model: str,         # e.g. "anthropic/claude-haiku-4-5" via OpenRouter
) -> int:
    """Returns phase index. Falls back to 'hypothesizing' (index 1) on failure."""
    
    # Build a compact transcript summary (last 3 turns max to save tokens)
    recent = transcript[-3:] if len(transcript) >= 3 else transcript
    transcript_text = "\n".join(
        f"{t['role'].upper()}: {t['content'][:200]}" for t in recent
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
        response = await client.complete(prompt, model=cheap_model, max_tokens=10)
        phase_word = response.strip().lower().split()[0]
        return PHASE_TO_IDX.get(phase_word, 1)  # default to hypothesizing
    except Exception:
        return 1  # safe fallback
```

### 3c. Upgrade the controller policy network

In `train_controller.py`, find the current policy network definition (the Torch MLP with actor-critic heads) and replace it with a GRU-based architecture:

```python
import torch
import torch.nn as nn

class ReasoningAwareController(nn.Module):
    """
    Controller with a GRU hidden state that tracks reasoning progress.
    
    Inputs per step:
        - state_features: the existing 28-dim feature vector
        - phase_onehot: 6-dim one-hot of current reasoning phase
    
    The GRU maintains h_t across steps within an episode.
    The policy head routes over the action space.
    The value head estimates returns for actor-critic baseline.
    The phase_probe head provides auxiliary supervision on the hidden state.
    """
    
    def __init__(self, state_dim: int = 28, n_phases: int = 6, 
                 n_actions: int = 6, hidden_size: int = 128):
        super().__init__()
        
        # Input to GRU: concat state features + phase onehot
        gru_input_dim = state_dim + n_phases
        
        # Single GRU cell — one step per agent call
        self.gru = nn.GRUCell(gru_input_dim, hidden_size)
        
        # Policy head: hidden state → action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        
        # Value head: hidden state → scalar value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Phase probe: auxiliary head to regularize hidden state
        # Predicts current phase from hidden state — trains alongside policy
        self.phase_probe = nn.Linear(hidden_size, n_phases)
        
        self.hidden_size = hidden_size
    
    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Zero hidden state at episode start."""
        return torch.zeros(batch_size, self.hidden_size)
    
    def forward(self, state_features: torch.Tensor, 
                phase_onehot: torch.Tensor,
                h_prev: torch.Tensor):
        """
        Args:
            state_features: (batch, 28)
            phase_onehot:   (batch, 6)
            h_prev:         (batch, hidden_size)
        Returns:
            action_logits, value, h_t, phase_logits
        """
        x = torch.cat([state_features, phase_onehot], dim=-1)  # (batch, 34)
        h_t = self.gru(x, h_prev)                               # (batch, hidden_size)
        
        action_logits = self.policy_head(h_t)   # (batch, n_actions)
        value = self.value_head(h_t).squeeze(-1) # (batch,)
        phase_logits = self.phase_probe(h_t)     # (batch, n_phases)
        
        return action_logits, value, h_t, phase_logits
```

Replace the old model instantiation in `train_controller.py` with:
```python
model = ReasoningAwareController(
    state_dim=28,       # existing feature vector size — do not change
    n_phases=6,
    n_actions=6,        # existing action space size — do not change  
    hidden_size=args.hidden_size,  # default 128, keep existing arg
)
```

### 3d. Update the episode rollout to thread the GRU hidden state

In the rollout/episode loop in `adaptive.py` or `train_controller.py`, make these changes:

```python
# At episode start:
h = model.initial_hidden(batch_size=1)

# At each decision step t:
phase_idx = await classify_reasoning_phase(transcript, client, cheap_model)
phase_onehot = torch.zeros(1, N_PHASES)
phase_onehot[0, phase_idx] = 1.0

state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)

action_logits, value, h, phase_logits = model(state_tensor, phase_onehot, h)
# h is now carried forward to the next step — do not reset within an episode

# Store for loss computation:
step_data.append({
    "action_logits": action_logits,
    "value": value,
    "phase_logits": phase_logits,
    "phase_label": torch.tensor([phase_idx], dtype=torch.long),
    "action_taken": action_idx,
    "log_prob": log_prob,
})
```

**Important:** Reset `h` to zeros at the start of each new episode, not between steps.

---

## Task 4: Replace Actor-Critic Loss with GRPO + Auxiliary Phase Loss

**File:** `arch-2/implementation/committee_llm/train_controller.py`

### 4a. Add GRPO loss function

Add this function:

```python
def grpo_loss(
    trajectories: list[dict],   # list of rollout dicts, each from same question
    rewards: list[float],        # verifiable reward for each trajectory
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    Group Relative Policy Optimization.
    Normalizes rewards within the group of trajectories for one question.
    Replaces standard REINFORCE baseline and actor-critic value loss.
    """
    if len(rewards) < 2:
        # Can't normalize a group of 1 — fall back to zero-centered
        advantages = [r - 0.5 for r in rewards]
    else:
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_r = max(std_r, 1e-8)
        advantages = [(r - mean_r) / std_r for r in rewards]
    
    policy_loss = torch.tensor(0.0)
    entropy_loss = torch.tensor(0.0)
    
    for traj, adv in zip(trajectories, advantages):
        adv_tensor = torch.tensor(adv, dtype=torch.float32)
        for step in traj["steps"]:
            # Policy gradient
            policy_loss = policy_loss - step["log_prob"] * adv_tensor
            # Entropy bonus to encourage exploration
            probs = torch.softmax(step["action_logits"], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropy_loss = entropy_loss - entropy_coef * entropy
    
    return policy_loss + entropy_loss


def auxiliary_phase_loss(
    trajectories: list[dict],
    phase_loss_weight: float = 0.1,
) -> torch.Tensor:
    """
    Auxiliary cross-entropy loss on the phase probe head.
    Regularizes the GRU hidden state to track reasoning phase.
    Weight is kept small so it doesn't dominate the policy signal.
    """
    loss = torch.tensor(0.0)
    criterion = nn.CrossEntropyLoss()
    
    for traj in trajectories:
        for step in traj["steps"]:
            loss = loss + criterion(step["phase_logits"], step["phase_label"])
    
    return phase_loss_weight * loss
```

### 4b. Update the training loop

Find the main training loop and replace the loss computation block:

```python
# OLD — remove:
# actor_loss = ...
# critic_loss = ...
# total_loss = actor_loss + 0.5 * critic_loss + entropy_bonus

# NEW:
# For each question, sample G=4 trajectories (or as many as budget allows)
# G=4 is enough at student budget; G=8 if you have headroom

G = 4  # group size for GRPO

for batch in training_batches:
    question = batch["question"]
    gold = batch["gold_answer"]
    benchmark = batch["benchmark"]
    
    # Sample G rollouts for this question
    trajectories = []
    rewards = []
    for _ in range(G):
        traj = await run_episode(question, model)
        r = compute_verifiable_reward(traj["final_answer"], gold, benchmark)
        trajectories.append(traj)
        rewards.append(r)
    
    # Compute losses
    pg_loss = grpo_loss(trajectories, rewards, entropy_coef=0.01)
    phase_loss = auxiliary_phase_loss(trajectories, phase_loss_weight=0.1)
    total_loss = pg_loss + phase_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 4c. Remove value head from loss

The GRPO formulation does not use a value baseline — reward normalization within the group replaces it. The value head on the model can stay (it's useful for analysis/logging) but **remove the critic loss term** from the training loop. Do not backprop through the value head during GRPO training.

### 4d. Update training config args

In `train_controller.py` argument parsing, add:

```python
parser.add_argument("--grpo-group-size", type=int, default=4,
    help="Number of trajectories to sample per question for GRPO normalization")
parser.add_argument("--phase-loss-weight", type=float, default=0.1,
    help="Weight for auxiliary phase classification loss")
parser.add_argument("--entropy-coef", type=float, default=0.01,
    help="Entropy bonus coefficient for exploration")
parser.add_argument("--cheap-model", type=str, 
    default="anthropic/claude-haiku-4-5-20251001",
    help="Model used for phase classification (should be cheapest available)")
```

Remove any args that were specific to the shaped reward (token penalty weights, latency weights, etc.).

---

## Task 5: Update Logging to Track New Signals

**File:** `arch-2/implementation/committee_llm/train_controller.py`

In `train_log.jsonl` and `eval_log.jsonl`, add these fields per episode/step:

```python
log_entry = {
    # existing fields — keep all of these
    "episode": episode_num,
    "reward": reward,
    "actions": action_sequence,
    
    # NEW fields to add
    "phase_sequence": phase_sequence,       # list of phase labels per step e.g. ["decomposing", "conflicting", "converging"]
    "phase_loss": phase_loss.item(),        # auxiliary loss value
    "grpo_advantages": advantages,          # normalized advantages for this group
    "group_rewards": rewards,               # raw rewards for all G trajectories in group
    "group_reward_mean": mean(rewards),     # mean reward within group
    "group_reward_std": std(rewards),       # std within group (low std = policy collapsed)
}
```

The `phase_sequence` field is critical — this is what you will analyze in your paper to show what routing behaviors the policy learned per task type.

---

## Task 6: Update Config File

**File:** `arch-2/implementation/configs/qwen3_8b_openrouter_arch2_controller.json`

Make these changes to the JSON:

```json
{
  "architecture": "arch2",
  "description": "Arch2: frozen specialist committee + reasoning-aware learned controller (GRPO + RLVR)",
  
  "controller": {
    "hidden_size": 128,
    "state_dim": 28,
    "n_phases": 6,
    "n_actions": 6
  },
  
  "training": {
    "grpo_group_size": 4,
    "phase_loss_weight": 0.1,
    "entropy_coef": 0.01,
    "lr": 3e-4,
    "grad_clip": 1.0
  },
  
  "reward": {
    "type": "verifiable",
    "hotpotqa_f1_threshold": 0.6,
    "gpqa_match": "letter_exact"
  },
  
  "models": {
    "committee": "qwen/qwen3-8b",
    "phase_classifier": "anthropic/claude-haiku-4-5-20251001"
  }
  
  // remove all shaped reward weight fields
}
```

---

## Task 7: Add Arch2-Specific Tests

Create `arch-2/tests/` directory with:

**`arch-2/tests/test_verifiable_reward.py`**
```python
"""Tests for RLVR reward function — no shaped components."""
from committee_llm.adaptive import compute_verifiable_reward

def test_gpqa_exact_match():
    assert compute_verifiable_reward("A", "A", "gpqa") == 1.0
    assert compute_verifiable_reward("B", "A", "gpqa") == 0.0

def test_gpqa_letter_extraction():
    # Model often outputs "A) Photosynthesis" — should still match
    assert compute_verifiable_reward("A) Photosynthesis occurs in chloroplasts", "A", "gpqa") == 1.0

def test_hotpotqa_exact():
    assert compute_verifiable_reward("Marie Curie", "Marie Curie", "hotpotqa") == 1.0

def test_hotpotqa_partial_f1():
    r = compute_verifiable_reward("Marie Curie was a physicist", "Marie Curie", "hotpotqa")
    assert 0.0 < r < 1.0

def test_hotpotqa_wrong():
    assert compute_verifiable_reward("Albert Einstein", "Marie Curie", "hotpotqa") == 0.0
```

**`arch-2/tests/test_reasoning_phase.py`**
```python
"""Tests for phase classifier and GRU controller."""
import torch
from committee_llm.train_controller import ReasoningAwareController
from committee_llm.adaptive import PHASE_TO_IDX, N_PHASES

def test_controller_forward_shape():
    model = ReasoningAwareController(state_dim=28, n_phases=6, n_actions=6, hidden_size=128)
    state = torch.zeros(1, 28)
    phase = torch.zeros(1, 6); phase[0, 0] = 1.0
    h = model.initial_hidden()
    action_logits, value, h_new, phase_logits = model(state, phase, h)
    assert action_logits.shape == (1, 6)
    assert h_new.shape == (1, 128)
    assert phase_logits.shape == (1, 6)

def test_hidden_state_updates():
    model = ReasoningAwareController()
    h0 = model.initial_hidden()
    state = torch.randn(1, 28)
    phase = torch.zeros(1, 6); phase[0, 2] = 1.0
    _, _, h1, _ = model(state, phase, h0)
    # Hidden state must change after a step
    assert not torch.allclose(h0, h1)

def test_phase_taxonomy_complete():
    assert N_PHASES == 6
    assert "stuck" in PHASE_TO_IDX
    assert "converging" in PHASE_TO_IDX
```

---

## Summary of All Files Changed

| File | Change Type |
|------|-------------|
| `committee_llm/config.py` | Housekeeping: Arch1 → Arch2 strings |
| `committee_llm/benchmark.py` | Housekeeping: parser description |
| `committee_llm/orchestrator.py` | Housekeeping: trace string |
| `committee_llm/adaptive.py` | Core: shaped reward → RLVR; add phase classifier; add phase/GRU state to episode loop |
| `committee_llm/train_controller.py` | Core: new ReasoningAwareController model; GRPO + auxiliary loss; updated logging; new CLI args |
| `configs/qwen3_8b_openrouter_arch2_controller.json` | Config: remove shaped reward fields, add GRPO/phase fields |
| `tests/test_verifiable_reward.py` | New: reward unit tests |
| `tests/test_reasoning_phase.py` | New: controller/phase unit tests |

## Do NOT Change

- `arch-2/implementation/committee_llm/tasks.py` — task schema is fine as-is
- `arch-2/implementation/committee_llm/client.py` — LLM client is unchanged
- `arch-2/implementation/committee_llm/evaluation.py` — eval harness is unchanged
- `arch-2/evals/data/benchmarks/` — benchmark data is unchanged
- `arch-2/docs/` — update manually after code changes are verified
- Anything under `arch-1/`