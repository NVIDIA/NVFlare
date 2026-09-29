# Agent Skills: Federated Statistics for Image Data

This example demonstrates using the NVFLARE Federated Statistics Agent Skill
with Codex or Claude Code. The coding agent creates and validates an NVFLARE
image-statistics job from per-site synthetic PNG files.

## Setup

Install NVFLARE using the [Installation guide](https://nvflare.readthedocs.io/en/main/installation.html), keeping
the stable, nightly, or editable distribution already selected. Then install the example dependencies:

```bash
python -m pip install -r requirements.txt
```

For an example downloaded with `nvflare examples get`, install the skills from its recorded source revision:

```bash
NVFLARE_REVISION=$(nvflare examples revision)
npx skills add "https://github.com/NVIDIA/NVFlare/tree/${NVFLARE_REVISION}/skills" --skill '*' -a codex -a claude-code -y
```

If you already have NVFLARE cloned at the same revision, use its top-level skills directory instead:

```bash
npx skills add "<nvflare-repo>/skills" --skill '*' -a codex -a claude-code -y
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
