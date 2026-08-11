# Agent Skills: Federated Statistics for Tabular Data

This example demonstrates using the NVFLARE Federated Statistics Agent Skill
with Codex or Claude Code. The coding agent creates and validates an NVFLARE
statistics job from per-site synthetic CSV files.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r requirements.txt
npx skills add ../../../../skills --skill '*' -a codex -a claude-code -y
```

## Code Structure

```text
fedstats-tabular/
├── README.md
├── generate_data.py        # creates deterministic per-site CSV files
├── requirements.txt
└── data/                   # generated site-1/ and site-2/ CSV inputs
```

## Data

Generate two site-local synthetic CSV files. Each site has 110 rows so a
10-bin histogram is valid with the recipe's default data-cleaning rules.

```bash
python generate_data.py
```

## Run Agent Workflow

Open this directory in Codex or Claude Code and use this prompt:

```text
I have tabular data from multiple sites in ./data. Calculate federated
statistics for it and validate the result locally.
```

## Output Summary

The coding agent reports aggregate statistics and the result location. Review
the results and the proposed changes before using the workflow with real data.
