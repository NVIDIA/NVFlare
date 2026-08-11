# NVFLARE Agent Skills: Runnable Examples

These examples are starting projects for a coding agent, not pre-built
NVFLARE jobs. Each directory contains an intentionally standalone workload,
an exact prompt, and the expected artifacts after the agent applies the named
NVFLARE skill. This makes the skill behavior reviewable before using it with a
real project or data.

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
its README. The prompt names the intended skill explicitly so the example is
repeatable. You may instead describe the same intent in your own words after
you are familiar with the workflow.

| Example | Skill | Starting workload |
| --- | --- | --- |
| [pytorch-conversion](pytorch-conversion) | `nvflare-convert-pytorch` | Plain PyTorch training loop |
| [lightning-conversion](lightning-conversion) | `nvflare-convert-lightning` | PyTorch Lightning `Trainer` |
| [huggingface-conversion](huggingface-conversion) | `nvflare-convert-huggingface` | Transformers `Trainer` |
| [fedstats-tabular](fedstats-tabular) | `nvflare-fed-stats` | Per-site CSV files |
| [fedstats-image](fedstats-image) | `nvflare-fed-stats` | Per-site PNG files |

All data in these directories is synthetic. The conversion skills create or
update `client.py` and `job.py`; review the diff before accepting it. Do not
use generated synthetic partitions, simulation settings, or privacy defaults
as an approval to apply the same choices to production data.
