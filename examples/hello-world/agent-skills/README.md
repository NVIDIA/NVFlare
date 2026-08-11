# NVFLARE Agent Skills: Runnable Examples

These examples are starting projects for a coding agent, not pre-built
NVFLARE jobs. Each directory contains an intentionally standalone workload
and a simple, goal-oriented prompt. This makes the agent-assisted behavior
easy to try before using it with a real project or data.

## One-time setup

From the NVFLARE repository root, install the complete skill set for the
coding agents you use:

```bash
npx skills add ./skills --skill '*' -a codex -a claude-code -y
```

The generated jobs require NVFLARE 2.9.0 or later. Install it in the Python
environment that the coding agent will use (or use an editable NVFLARE
checkout):

```bash
python -m pip install 'nvflare~=2.9.0'
```

Open one example directory in Codex or Claude Code, then paste the prompt in
its README. The coding agent selects the appropriate NVFLARE skill from the
intent and source material; users do not need to know skill names.

| Example | Workflow | Starting workload |
| --- | --- | --- |
| [pytorch-conversion](pytorch-conversion) | Convert a plain PyTorch loop | Plain PyTorch training loop |
| [lightning-conversion](lightning-conversion) | Convert a Lightning `Trainer` | PyTorch Lightning `Trainer` |
| [huggingface-conversion](huggingface-conversion) | Convert a Transformers `Trainer` | Transformers `Trainer` |
| [fedstats-tabular](fedstats-tabular) | Compute federated statistics | Per-site CSV files |
| [fedstats-image](fedstats-image) | Compute image statistics | Per-site PNG files |

All data in these directories is synthetic. Review the agent's proposed changes
before accepting them. Do not use generated simulation settings or privacy
defaults as an approval to apply the same choices to production data.
