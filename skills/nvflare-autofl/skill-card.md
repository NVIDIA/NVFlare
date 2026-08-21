## Description: <br>
Use for agent-assisted Auto-FL optimization of an existing NVFLARE job in simulation, POC, or production. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to iteratively optimize NVFLARE federated learning jobs through isolated, reproducible candidate campaigns that preserve metric semantics and comparison budgets. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Job Import Contract](references/job-import-contract.md) <br>
- [Continuous Campaigns](references/continuous-campaigns.md) <br>
- [Experiment Comparability](references/experiment-comparability.md) <br>
- [Bounded Campaign Example](references/bounded-campaign-example.md) <br>
- [NVFlare Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVFlare Paper (arXiv:2210.13291)](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [JSON, Configuration files, Shell commands, Analysis] <br>
**Output Format:** [JSON envelopes, TSV ledger, YAML configuration, and Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive) from skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helps the agent complete the user's goal and expected workflow. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 58% → 77% (+20 points) | 48% → 70% (+22 points) |
| Security | 100% → 100% (±0 points) | 71% → 86% (+14 points) |
| Correctness | 57% → 86% (+29 points) | 57% → 74% (+17 points) |
| Discoverability | 46% → 77% (+31 points) | 39% → 74% (+35 points) |
| Effectiveness | 45% → 54% (+9 points) | 38% → 48% (+10 points) |
| Efficiency | 39% → 70% (+31 points) | 36% → 69% (+32 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
