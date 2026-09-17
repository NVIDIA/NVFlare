# Agent Skills: Federated Statistics for Image Data

This example demonstrates using the NVFLARE Federated Statistics Agent Skill
with Codex or Claude Code. The coding agent creates and validates an NVFLARE
image-statistics job from per-site synthetic PNG files.

## Setup

Install NVFLARE using the [Installation guide](https://nvflare.readthedocs.io/en/main/installation.html), keeping
the stable, nightly, or editable distribution already selected. Then install the example dependencies and the skills
from the same source revision as this example:

```bash
python -m pip install -r requirements.txt
if [ -f .nvflare-example.json ]; then
  NVFLARE_REVISION=$(nvflare --version revision)
  NVFLARE_SKILLS_SOURCE="https://github.com/NVIDIA/NVFlare/tree/${NVFLARE_REVISION}/skills"
else
  NVFLARE_SKILLS_SOURCE=../../../../skills
fi
npx skills add "$NVFLARE_SKILLS_SOURCE" --skill '*' -a codex -a claude-code -y
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
