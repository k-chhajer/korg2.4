# Architecture 1

Architecture 1 is the explicit-reference control for the paper. It is not the novelty claim.

Required definition:

- four separate instances of the same base model
- roles: coordinator, researcher, analyst, critic
- fixed topology: coordinator -> researcher -> analyst -> critic -> coordinator
- separate system prompt per role
- separate role-specific output schema per role
- separate local context view per role
- separate inference call per role
- specialization must come from role-specific post-training, not only prompt wording

What the code enforces now:

- fixed stage sequence in [committee_llm/config.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/config.py)
- stage execution and schema validation in [committee_llm/orchestrator.py](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/committee_llm/orchestrator.py)
- role prompts in [implementation/prompts](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/prompts)
- role metadata, output schemas, and local context views in config
- paper-claim blocker detection when a config is only a prompt scaffold

What counts as paper-valid:

- `architecture.specialization_source = post_trained_role_specialists`
- each role has its own checkpoint identifier
- no checkpoint reuse across roles
- the fixed coordinator -> researcher -> analyst -> critic -> coordinator route is preserved

What does not count as paper-valid:

- one shared `qwen3:8b` or `Qwen/Qwen3-8B-Instruct` endpoint with only role prompts changed
- calling the scaffold a post-trained committee without actual role-tuned checkpoints

Configs:

- local scaffold: [qwen3_8b_ollama_eval.json](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/configs/qwen3_8b_ollama_eval.json)
- paper template: [qwen3_arch1_posttrained_template.json](C:/Users/luthi/Documents/korg/korg2_4_remote/arch-1/implementation/configs/qwen3_arch1_posttrained_template.json)
