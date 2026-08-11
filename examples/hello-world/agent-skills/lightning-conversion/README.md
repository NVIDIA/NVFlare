# Agent Skills: Convert a PyTorch Lightning Project

This example demonstrates using the NVFLARE Convert Lightning Agent Skill with
Codex or Claude Code. The starting project is a standalone Lightning
`Trainer` workflow that the coding agent converts to a two-site NVFLARE FedAvg
job.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r source/requirements.txt
npx skills add ../../../../skills --skill '*' -a codex -a claude-code -y
```

## Code Structure

```text
lightning-conversion/
├── README.md
└── source/
    ├── model.py            # standalone LightningModule
    ├── train.py            # standalone Trainer workflow
    └── requirements.txt    # source-project dependencies
```

## Data

The example creates deterministic synthetic tensors for its train and
validation data loaders. The converted simulation must use site-local data.

## Model

`source/model.py` defines `LitClassifier`, including its optimizer,
training step, and validation accuracy metric.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
I have an existing PyTorch Lightning training project in ./source. Convert it
to federated learning and validate it locally.
```

## Run the Starting Project

Run the starting standalone project with:

```bash
python source/train.py
```

## Output Summary

The coding agent reports the validation result and artifact paths. Review its
changes before using the pattern with real data.
