## Description: <br>
Use for agent-assisted Auto-FL optimization of an existing NVFLARE job in simulation, POC, or production. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to optimize federated learning jobs built with NVIDIA FLARE through an automated campaign of isolated, reproducible candidate experiments across simulation, POC, and production environments. <br>

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


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [JSON envelopes, YAML configuration, TSV ledgers, and Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), each attempt ran in an isolated sandbox pod with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether it is safe to use; scored by the security signal (100%). <br>
- Correctness: Checks whether the answer is correct; scored by the accuracy signal (100%). <br>
- Discoverability: Checks whether the right skill was loaded when needed; scored by the skill_execution signal (100%). <br>
- Effectiveness: Checks whether the skill helped complete the task; scored by goal_accuracy (50%) and behavior_check (50%). <br>
- Efficiency: Checks whether wasted tool calls and token usage were avoided; scored by skill_efficiency (50%) and token_efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `skill_execution`: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `token_efficiency`: Measures actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 74.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 81.8% → 100.0% (+18.2 points) |
| Correctness | Not available | 67.3% → 72.5% (+5.2 points) |
| Discoverability | Not available | 63.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 32.6% → 44.0% (+11.4 points) |
| Efficiency | Not available | 90.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
