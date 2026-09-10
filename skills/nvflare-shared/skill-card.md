## Description: <br>
Internal NVFLARE conversion references and templates used only when another NVFLARE skill directs the agent to a shared workflow, policy, or asset. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building federated learning workflows use this shared reference library for NVFLARE conversion policies, templates, and guidance. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [conversion-common.md](references/conversion-common.md) <br>
- [conversion-workflow.md](references/conversion-workflow.md) <br>
- [dependency-install.md](references/dependency-install.md) <br>
- [metrics-and-artifact-reporting.md](references/metrics-and-artifact-reporting.md) <br>
- [pytorch-family-recipe-construction.md](references/pytorch-family-recipe-construction.md) <br>
- [pytorch-family-recipe-selection.md](references/pytorch-family-recipe-selection.md) <br>
- [pytorch-model-exchange.md](references/pytorch-model-exchange.md) <br>
- [runtime-output-guidance.md](references/runtime-output-guidance.md) <br>
- [site-data-and-paths.md](references/site-data-and-paths.md) <br>
- [validation-evidence.md](references/validation-evidence.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Code templates] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (3 positive) with 3 attempts per task across 2 agents in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% behavior check). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 82.1% | 78.3% |
| Security | 83.3% → 83.3% (±0.0 points) | 33.3% → 66.7% (+33.4 points) |
| Correctness | 86.7% → 100.0% (+13.3 points) | 93.3% → 100.0% (+6.7 points) |
| Discoverability | 71.7% | 63.3% |
| Effectiveness | 75.0% → 78.3% (+3.3 points) | 73.3% → 88.3% (+15.0 points) |
| Efficiency | 77.3% | 73.0% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
