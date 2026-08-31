## Description: <br>
Use for agent-assisted Auto-FL optimization of an existing NVFLARE job in simulation, POC, or production. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to iteratively optimize NVFLARE federated learning jobs through agent-assisted campaigns that propose, evaluate, and track candidate changes against a measured objective. <br>

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
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis, Configuration instructions, Files] <br>
**Output Format:** [JSON envelopes with Markdown summaries and TSV ledger] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Persists autofl.yaml, results.tsv, campaign_state.json, candidate manifests, run artifacts, and progress plots] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 7 evaluation tasks (7 positive) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 60% → 73% (+13 points) |
| Security | 86% → 100% (+14 points) |
| Correctness | 71% → 71% (±0 points) |
| Discoverability | 38% → 71% (+33 points) |
| Effectiveness | 64% → 55% (-9 points) |
| Efficiency | 41% → 69% (+29 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
