# Benchmark Comparison Summary

## HotpotQA Distractor

Same first-75 HotpotQA examples, Qwen3-8B-AWQ on AWS g4dn.

| System | Completed | Failed | Answer EM | Answer F1 | Mean sec/task | Model calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal Qwen single-shot | 75 | 0 | 0.000 | 0.016 | 23.1 | 75 | 174,866 |
| Thinking Qwen single-shot | 75 | 0 | 0.000 | 0.016 | 21.6 | 75 | 174,866 |
| Committee | 73 | 2 | 0.384 | 0.577 | 150.0 | 365 | 1,543,424 |

On the 73 matched completed tasks, committee won answer F1 on 68 tasks, normal Qwen won on 4, and 1 tied. Committee was much stronger on answer quality, but used about 9.2x tokens and 6.5x latency versus normal single-shot.

The April 15 single-shot thinking run used `chat_template_kwargs.enable_thinking=true` and sampled traces begin with `<think>`, confirming the Qwen3 thinking path was active. Its aggregate HotpotQA answer metrics and token totals matched the non-thinking single-shot run on this 75-example slice.

Artifacts:
- `arch-1/evals/runs/hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/summary.json`
- `s3://korg24-arch1-gpqa-20260406-525184038455/hotpotqa_aws_g4dn_qwen_single_shot_thinking_75_20260415/summary.json`
- `arch-1/evals/runs/hotpotqa_aws_g4dn_thinking_75_20260407_retry4b/summary.json`

## GPQA Diamond

Full 198-example committee run, Modal-hosted Qwen3-8B. The run hit the Modal billing cap before all examples completed.

| System / Reference | Completed | Failed | Accuracy | Mean sec/task | Model calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Committee, completed only | 163 | 35 | 51.53% | 383.2 | 815 | 3,361,018 |
| Committee, full-set lower bound | 198 | 35 | 42.42% | - | 815 | 3,361,018 |
| Qwen3-8B non-thinking official | 198 | - | 39.3% | - | - | - |
| Qwen3-8B thinking official | 198 | - | 62.0% | - | - | - |

Committee beat the official non-thinking Qwen3-8B reference on completed traces and on the conservative full-set lower bound. It remained below the official thinking Qwen3-8B reference.

Artifact:
- `arch-1/evals/reports/gpqa_diamond_modal_committee_198_20260405.md`
