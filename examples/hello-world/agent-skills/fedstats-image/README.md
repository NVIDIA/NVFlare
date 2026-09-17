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
I have image data from multiple sites in ./data. Calculate federated image
statistics for it and validate the result locally.
```

## Output Summary

The coding agent reports aggregate image statistics and the result location.
Review the results and the proposed changes before using the workflow with real
data.
