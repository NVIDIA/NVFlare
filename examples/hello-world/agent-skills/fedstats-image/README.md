# Agent Skills: Federated Statistics for Image Data

This example demonstrates using the NVFLARE Federated Statistics Agent Skill
with Codex or Claude Code. The coding agent creates and validates an NVFLARE
image-statistics job from per-site synthetic PNG files.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r requirements.txt
npx skills add ../../../../skills --skill '*' -a codex -a claude-code -y
```

## Code Structure

```text
fedstats-image/
├── README.md
├── generate_data.py        # creates deterministic per-site PNG files
├── requirements.txt
└── data/                   # generated site-1/ and site-2/ image inputs
```

The coding agent adds `client.py` and `job.py` during the workflow.

## Data

Generate the synthetic image dataset before asking the agent to analyze it.
The generator creates 110 images per site, sufficient for the requested
10-bin histogram, plus one intentionally corrupt file to exercise
`failure_count`.

```bash
python generate_data.py
```

## Run Agent Workflow

Open this directory in Codex or Claude Code and use this prompt:

```text
Compute federated image count, failure count, and a 10-bin pixel-intensity
histogram across ./data. Treat site-1 and site-2 as separate sites and do not
move image files or pixels between sites. Create client.py and job.py in this
directory, run the generated job in local simulation, and report aggregate
results and the output JSON path.
```

## Run Job

After reviewing the generated files, run the statistics simulation:

```bash
python job.py
```

## Output Summary

The generated job reports only aggregate image counts, failure counts, and
pixel-intensity histograms. It writes a result JSON at the path reported by the
coding agent.
