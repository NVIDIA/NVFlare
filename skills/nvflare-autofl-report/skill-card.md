## Description: <br>
Generate a reproducible final report, literature-outcome synthesis, JSON summary, and refreshed progress plot for a stopped or interrupted NVFLARE Auto-FL campaign. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to produce final reports, literature-outcome syntheses, JSON summaries, and refreshed progress plots for stopped or interrupted NVFLARE Auto-FL federated learning campaigns without modifying campaign evidence. <br>

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
- [Auto-FL Final Report Contract](references/report-contract.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE GitHub](https://github.com/NVIDIA/NVFlare) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Files] <br>
**Output Format:** [Markdown report, JSON summary, and optional PNG progress plot] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive) covering stopped-campaign reporting plus active, global, and optimize negative scenarios, with 1 attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (scored from the `security` signal). <br>
- Correctness: Whether the answer is correct (scored from the `accuracy` signal). <br>
- Discoverability: Whether the right skill was loaded when needed (scored from the `skill_execution` signal). <br>
- Effectiveness: Whether the skill helped complete the task (50% `goal_accuracy` + 50% `behavior_check`). <br>
- Efficiency: Whether it avoided wasted tool calls and token usage (50% `skill_efficiency` + 50% `token_efficiency`). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.7% — baseline ran, but no comparable score was available; uplift unavailable | 70.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 75.0% (±0.0 points) | 75.0% → 75.0% (±0.0 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 45.0% → 80.0% (+35.0 points) |
| Discoverability | 73.8% — baseline ran, but no comparable score was available; uplift unavailable | 61.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 75.0% (+25.0 points) | 41.9% → 47.5% (+5.6 points) |
| Efficiency | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
