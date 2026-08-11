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

The coding agent adds `client.py` and `job.py` during the workflow.

## Data

Generate two site-local synthetic CSV files. Each site has 110 rows so a
10-bin histogram is valid with the recipe's default data-cleaning rules.

```bash
python generate_data.py
```

## Run Agent Workflow

Open this directory in Codex or Claude Code and use this prompt:

```text
Use the installed nvflare-fed-stats skill to compute federated count, mean,
standard deviation, and a 10-bin histogram for every numeric feature in
./data. Treat site-1 and site-2 as separate sites; do not combine raw rows.
Create client.py and job.py in this directory, run the generated job in local
simulation, and report only aggregate results and the output JSON path.
```

## Run Job

After reviewing the generated files, run the statistics simulation:

```bash
python job.py
```

## Output Summary

The generated job reports aggregate count, mean, standard deviation, and
histogram results only. It retains the FedStats privacy cleaners and writes the
result JSON to the path reported by the coding agent.
