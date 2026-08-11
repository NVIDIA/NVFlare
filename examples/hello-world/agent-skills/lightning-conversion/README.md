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
npx skills add ../../../../skills -a codex -a claude-code
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

The coding agent adds `source/client.py` and `source/job.py` during the
conversion.

## Data

The example creates deterministic synthetic tensors for its train and
validation data loaders. The converted simulation must use site-local data.

## Model

`source/model.py` defines `LitClassifier`, including its optimizer,
training step, and validation accuracy metric.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
Use the installed nvflare-convert-lightning skill to convert ./source into a
two-site NVFLARE FedAvg job. Preserve the LightningModule, validation metric,
and one-epoch local training budget. Use deterministic site-local synthetic
data for simulation, create client.py and job.py in ./source, validate the
exported job locally, and summarize the result.
```

## Run Job

Run the starting standalone project with:

```bash
python source/train.py
```

After reviewing the agent-generated files, run the federated simulation:

```bash
cd source
python job.py
```

## Output Summary

A successful conversion retains the application-owned `LightningModule` and
uses the Lightning Client API patch for federated exchange. The coding agent
reports the generated files, validation result, and artifact paths.
