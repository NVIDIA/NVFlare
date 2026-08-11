# Agent Skills: Convert a Plain PyTorch Project

This example demonstrates using the NVFLARE Convert PyTorch Agent Skill with
Codex or Claude Code. The starting project is a standalone CPU-only PyTorch
classifier; the coding agent converts it to a two-site NVFLARE FedAvg job.
It is recommended to create a virtual environment before running the example.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).
Install NVFLARE 2.9.0 or later, the example dependencies, and the complete
Agent Skills package from an NVFLARE checkout:

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r source/requirements.txt
npx skills add ../../../../skills --skill '*' -a codex -a claude-code -y
```

## Code Structure

```text
pytorch-conversion/
├── README.md
└── source/
    ├── model.py            # standalone PyTorch model definition
    ├── train.py            # standalone train and evaluation loop
    └── requirements.txt    # source-project dependencies
```

After conversion, `source/client.py` and `source/job.py` are added by the
coding agent.

## Data

The standalone source creates deterministic synthetic feature and label
tensors. The prompt asks the skill to create deterministic, site-local
simulation partitions; it must not pool records across sites.

## Model

`source/model.py` defines `TinyClassifier`, a two-layer classifier. The
standalone `source/train.py` performs one local epoch and evaluates accuracy
with cross-entropy loss.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
Convert ./source into a two-site NVFLARE FedAvg job. Preserve the model,
cross-entropy evaluation, and one local training epoch. Generate deterministic,
site-local synthetic data for simulation; do not pool records across sites.
Create client.py and job.py in ./source, validate the exported job with a local
simulation, and summarize the changed files and validation result.
```

## Run Job

First, the original standalone program can be run with:

```bash
python source/train.py
```

After reviewing the generated `source/client.py` and `source/job.py`, run the
generated job from its directory:

```bash
cd source
python job.py
```

## Output Summary

The coding agent reports the simulation result and artifact paths. A successful
conversion preserves the evaluation metric and sends the actual completed
local optimizer-step count for FedAvg weighting. Review the generated files
before using the pattern with real data.
