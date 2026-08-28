## Description: <br>
Convert existing plain or manual PyTorch training code into an NVFLARE federated job using Client API model exchange, local validation, and job export. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers converting existing PyTorch training scripts into NVIDIA FLARE federated learning jobs using Client API model exchange, recipe-based aggregation, and job export. <br>

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
- [PyTorch Client API Conversion Reference](references/pytorch-client-api-conversion.md) <br>
- [Recipe Selection Reference](references/recipe-selection.md) <br>
- [Job Validation Reference](references/job-validation.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper (arXiv:2210.13291)](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Python files and shell commands with validation output] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 17 positive tasks in isolated sandbox pods, covering safety, correctness, discoverability, effectiveness, and efficiency. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use without unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Whether the generated conversion answer is correct against the reference. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage during execution. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Determines whether the user's goal was achieved. <br>
- `behavior_check`: Validates whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 48% → 68% (+21 points) |
| Security | Not available | 32% → 29% (-3 points) |
| Correctness | Not available | 85% → 94% (+9 points) |
| Discoverability | Not available | 26% → 68% (+41 points) |
| Effectiveness | Not available | 64% → 74% (+10 points) |
| Efficiency | Not available | 30% → 76% (+45 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
