grok 4.2

- four agents (captain, harper, benjamin, lucas) form a native, production multi-agent collaboration system that runs on every sufficiently complex query (not a user-facing framework you have to orchestrate like autoGen or swarm) but a baked-in inference-time architecture where four specialized replicas of the underlying model collaborate in real time
- agent roles include:
    - grok (captain): task decomposition, overall strategy, conflict resolution, final synthesis and delivery of a coherent answer
    - harper (research and facts): real-time search, data gathering (a lot of x data is used here), evidence integration, primary fact-verification
    - benjamin (math/code/logic): rigorours reasoning, programming, math proofs, stress-testing of strategies and logic chains
    - lucas (creativity/balance): divergent thinking, novel hypotheses, ux optimizaiton, adding human touch4

detailed plan:

What the Research Actually Confirmed
First, the architecture is now much clearer. It's confirmed as a ~3T parameter MoE backbone with ~500B active parameters, all four agents sharing the same weights and the same KV cache for the input context. The critique round is confirmed to be extremely short — about 180 tokens total across all agents — and it's not text debate, it's agents exchanging compressed critique embeddings through a lightweight cross-attention block that was added specifically in 4.20. The whole thing is trained end-to-end with what they describe as "pre-training-scale RL" for the orchestration layer.
The second critical thing the research confirmed: there's now a whole academic subfield that's directly trying to do what you want to do, and it has real published results. The papers are MAGRPO (Multi-Agent GRPO, August 2025) and Stronger-MAS / AT-GRPO (October 2025). These are the closest open-source replications of the training methodology xAI used. They exist, they have code, and they work on Qwen3.

The Closest Buildable Implementation
The closest thing you can actually build, which is both research-grade and mechanistically faithful to what xAI did, is this:
One shared base model. Role specialization through joint MARL training. Shared prefix serving. Structured embedding-level critique.
Not four separate fine-tuned models. One model that has been trained with multi-agent RL so that role-specific behavior is baked into the weights, not just prompted at inference time. This is the key distinction that makes your project genuinely novel rather than just "four API calls in parallel."
Here's how to do it concretely.

The Stack
Base model: Qwen3-8B or Qwen3-14B. This is the right choice for three reasons. First, Qwen3 already has strong reasoning capability in the 8B range due to its hybrid thinking mode. Second, verl (the RL training framework) has native Qwen3 support with tested configurations. Third, Qwen3's MoE variant (Qwen3-30B-A3B, 30B total but only 3B active) is actually a structurally faithful replication of the MoE approach xAI is using — you'd be training a MoE system where different expert clusters get reinforced for different roles, which is exactly H-02 from our earlier discussion.
Training framework: verl (Volcano Engine RL). Not alternatives — verl specifically. It's the framework that DAPO, the current SOTA RL algorithm for reasoning, was trained with. It supports GRPO, DAPO, multi-turn agentic RL, and critically it has async rollout support so your multi-agent RL training doesn't stall waiting for agent turns. The GitHub repo is verl-project/verl and it has native Qwen3 support as of mid-2025.
The RL algorithm: AT-GRPO (Agent-and-Turn-wise GRPO), which is the algorithm from Stronger-MAS. This is the multi-agent extension of GRPO that solves the credit assignment problem — when four agents collaborate and produce a final answer, how do you assign reward to each agent's turn individually? AT-GRPO handles this by grouping trajectories by agent role and turn, computing relative advantages within each group, and backpropagating role-specific gradient signals. This is conceptually what xAI almost certainly did, and it's open source and tested on Qwen models at the 1.7B and 8B scale.
Inference serving: vLLM with prefix caching enabled. For the shared input context, vLLM's prefix caching will automatically reuse the KV cache across all four agent decode streams if they share the same prefix. You configure this with enable_prefix_caching=True. You won't get xAI's exact shared-block implementation across a 200K GPU cluster, but you get the same semantic benefit at your scale: the input is computed once, all four agents branch from that cached state.

The Architecture Decision You Must Make First
There are two architecturally distinct approaches and the research has a direct comparison:
Role-sharing: One model instance, four different system prompt prefixes, all four agents are the same weights at different positions in the same forward pass. Specialization comes purely from prompting and RL reward shaping.
Role-specialized: Four separate fine-tuned models, each fine-tuned with role-specific reward signals. More compute to serve, but potentially stronger specialization.
The Stronger-MAS paper tested both on coding and math tasks and found that the right answer depends on task type. For tasks where the roles are genuinely different (a fact-checker and a mathematician doing very different things), role-specialized policies win. For tasks where the roles are variations on the same capability, role-sharing wins because it generalizes better.
For replicating Grok 4.20 specifically, xAI almost certainly uses role-sharing on a shared weight backbone (they said explicitly "four specialized replicas of the underlying model"), but with RL that has pushed the weight space into role-sensitive regions. This is subtly different from both pure role-sharing and pure role-specialized: it's one model that has been trained to produce genuinely different behavior conditional on the role token. Mechanistically this means the role token has become a steering vector that activates different behavioral circuits in the shared weights.
For your build, start with role-sharing on one Qwen3-8B. If the specialization gap is too small after RL training, switch to role-specialized LoRA adapters. LoRA adapters per role means you have one base model with four small adapter sets that swap in — you get role specialization without four full model instances. This is probably the most compute-efficient path.

Training Pipeline in Detail
Step 1: Synthetic multi-agent trace generation (before any RL)
Before you touch verl, you need to generate ~5,000 examples of multi-agent collaboration traces. Use GPT-4o or a strong Claude model as the generator. For each of 1,000 complex prompts across math, coding, factual reasoning, and analytical tasks:
Generate what an ideal Captain decomposition looks like. Generate what ideal Benjamin, Harper, and Lucas outputs look like given that decomposition. Generate what an ideal structured critique looks like from each agent. Generate what ideal Captain synthesis looks like given all of the above.
This synthetic dataset is your SFT warmup. You fine-tune your Qwen3-8B on these traces first to teach it the multi-agent protocol format before RL training. Without this step, early RL training is chaotic because the model has no prior for how agents are supposed to communicate. The SFT warmup gives you a sensible initialization for the RL phase.
Step 2: Role-conditioned reward model design
This is the most intellectually demanding component. You need separate reward signals per role because a Benjamin output should be judged on logical rigor, not creativity, and a Lucas output should be judged on divergent thinking, not mathematical precision.
For Benjamin, reward is verifiable: pass rate on unit tests for code, symbolic verification for math proofs, logical consistency checker for reasoning chains. All rule-based, no reward model needed. This is RLVR (RL with Verifiable Rewards) — the same approach DeepSeek used, and it's the most stable form of RL training.
For Harper, reward is factual grounding quality. You can approximate this with a retrieval-augmented verification step: take Harper's factual claims, retrieve supporting evidence via web search, score each claim as supported/unsupported/contradicted. This is a learned reward signal but you can make it mostly rule-based by treating factual grounding as a binary per-claim metric.
For Lucas, reward is the hard one. You can't do RLVR here because there's no verifiable ground truth for creativity. Two viable approaches: (1) train a small reward model on human preference data for creative/divergent outputs, or (2) use a diversity reward — Lucas gets rewarded for producing outputs that are measurably different from what Captain and Benjamin produce (measured by embedding distance). The second is fully self-supervised and avoids training a reward model entirely.
For Captain, the reward is the quality of the final output — whatever your downstream benchmark metric is. This propagates a signal back through the synthesis step. Captain learns to weight agent contributions in ways that maximize final output quality.
Step 3: AT-GRPO multi-agent RL training with verl
This is where you use the Stronger-MAS AT-GRPO implementation. The key configuration choices:
Set T=4 (four turns per episode, one per agent). The advantage is computed agent-and-turn-wise: for agent i at turn t, you compute the group-relative advantage over other rollouts of the same agent at the same turn with the same task. This ensures credit assignment is clean — Benjamin's gradient is only relative to other Benjamin outputs, not compared to Lucas's creative outputs.
For the reward mixing coefficient, start with equal weighting across roles and tune based on which roles are learning too fast vs. too slow. In Stronger-MAS experiments, math tasks needed higher Benjamin reward weight, which makes sense.
The training loop at each iteration: sample a batch of complex prompts, run the full 4-agent pipeline with your current model, collect the multi-agent trajectory (Captain decomposition → parallel agent outputs → critique round → synthesis), score each component with its role-appropriate reward, compute AT-GRPO advantages, backpropagate. The key verl configuration for this is setting multi_turn: true and configuring the agent rollout loop with your four-agent workflow.
Use LoRA for the initial RL training phase (rank 64 is enough) to reduce VRAM requirements and allow faster iteration. If your LoRA specialization gap is large enough (which you measure by per-role benchmark scores), you may not need full fine-tuning.
Step 4: The structured critique mechanism
The 180-token constraint from the confirmed architecture tells you something precise: the critique round is not a full agent response. It's highly compressed. Based on the cross-attention block description, the most faithful replication is agents exchanging not text but a structured object.
In your implementation, design the critique as a fixed-length structured output with three fields:
challenge_type: one of [factual, logical, missing_consideration, alternative_approach]
target: a ≤20-token quote of the exact claim being challenged
correction: a ≤50-token proposed correction or alternative
This caps each agent's critique at roughly 80 tokens and keeps the full critique round under 240 tokens. Train this format into the SFT warmup so agents know how to produce it. During RL training, include a critique quality reward: did the challenge target a real claim in the output? Did the proposed correction improve the final answer when incorporated?
The reason this matters beyond efficiency: structuring the critique forces agents to be precise rather than vague. Free-text debate lets agents say "this seems incorrect" without specifying what. Structured critique forces "claim X at position Y is factually wrong, replace with Z." That precision is what makes the critique round actionable for Captain.
Step 5: Captain synthesis training
Captain is a separate fine-tuning target from the role agents. After the RL training stabilizes the role agents, fine-tune Captain's behavior specifically on synthesis. Build a dataset of (4 agent outputs + structured critiques → gold synthesis) pairs, where the gold synthesis is generated by GPT-4o and then filtered for cases where it's genuinely better than any individual agent's output.
The two things to explicitly train Captain to do: (1) resolve contradictions when Harper and Benjamin disagree, by citing Harper's evidence or Benjamin's logical chain as the tiebreaker, and (2) attribute contributions by including specific reasoning from Benjamin when the answer has a mathematical component, rather than paraphrasing everything into the same generic style.

The Novel Research Contribution
Now that I've done the research, I can be more precise about what's actually novel here because the academic field is moving fast and I can see the gaps.
The gap in existing work — MAGRPO and Stronger-MAS both train multi-agent systems but on homogeneous roles. MAGRPO trains a coder-tester loop. Stronger-MAS trains coder-reviewer loops. Nobody has published a trained open-source system with genuinely heterogeneous roles (factual retrieval + mathematical reasoning + creative generation + synthesis) trained jointly with multi-agent RL. That's your contribution.
Specifically publishable: the comparison between (a) four agents trained jointly with AT-GRPO using role-specific rewards versus (b) four agents trained independently with single-agent GRPO using the same role-specific rewards versus (c) four prompted agents with no RL. The joint training hypothesis is that agents learn to communicate more efficiently when their communication protocol is also being RL-optimized, not just their individual outputs. This is the key question Stronger-MAS asks but doesn't fully answer for heterogeneous roles.
The second publishable result: compute-matched comparison. Your 4×8B system (32B total, 8B active at any one time given LoRA switching) compared to a single 70B model. If joint MARL training produces a system that matches 70B on the benchmarks using equivalent total compute, that's the result that makes this matter.

Papers to Read, in Actual Priority Order
Read before you touch verl:
Stronger-MAS (AT-GRPO) — arxiv 2510.11062. This is your training algorithm. Read the full paper, not just the abstract. The appendix has the exact workflow configurations for code and math tasks which you'll adapt.
MAGRPO — arxiv 2508.04652. The earlier complementary paper that frames multi-agent RL as Dec-POMDP. Gives you the theoretical framework for why joint training converges better than independent training.
Read before designing your reward functions:
DeepSeekMath (GRPO) — Shao et al. 2024. GRPO is the base algorithm everything else builds on. You need to understand it mechanically, not just know it exists.
DAPO — this is SOTA over GRPO for reasoning tasks, is fully open source, and is directly supported in verl. Understand how it improves over GRPO (clip-higher, dynamic sampling) because you might want to use DAPO rather than vanilla GRPO for Benjamin's math training.
Search-R1 or DeepResearch-style papers for Harper's retrieval reward design. Training a model to use web search as a tool with RL reward based on factual grounding is now well-established.
Read before the critique mechanism:
Irving & Amodei 2018, AI Safety via Debate. Not for safety — for the theoretical grounding that constrained critique between disagreeing agents produces more truthful outputs than unconstrained generation. Your 180-token critique constraint is a practical instantiation of this.
Read before you design your benchmarks:
Stronger-MAS appendix C.2 specifically — they document their multi-agent workflow configurations for different task types in detail, which will save you weeks of trial and error.

Benchmarks and Realistic Targets
With Qwen3-8B trained with AT-GRPO joint multi-agent RL, compared to a single Qwen3-72B as the compute-matched baseline:
MATH-500: Qwen3-72B single-agent sits around 85-88%. Your 4×8B target is 82-87%. You're not dramatically beating it but you're matching it at a fraction of the active parameter cost.
GPQA Diamond: This is where multi-agent genuinely helps because it requires both factual retrieval (Harper) and rigorous reasoning (Benjamin) simultaneously. Qwen3-72B is around 65-68%. Your 4×8B target is 62-70% — potentially matching the much larger model.
LiveCodeBench: Benjamin-heavy task. Qwen3-72B around 65%. Your target is 60-68%.
GAIA (web search + multi-hop reasoning): This is your strongest benchmark because GAIA specifically rewards the kind of factual grounding + synthesis that your system is designed for. A Harper agent with real web search tool use trained via RL should perform significantly better here than a frozen model. Target: matching GPT-4o-level GAIA performance with your 8B system.
The comparison that makes the paper: plot a curve of quality vs. active parameters for (single model, various sizes) and overlay your multi-agent system. If your system sits above the single-model scaling curve — equal quality at lower active parameter count — you have demonstrated that collaborative architecture is a qualitatively better use of parameters than scale alone. That's the result.
