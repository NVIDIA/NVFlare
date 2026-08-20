# Agent Skills: Convert a Plain PyTorch Project

This example demonstrates using the NVFLARE Convert PyTorch Agent Skill with
Codex or Claude Code. The starting project is a standalone, CPU-first PyTorch
classifier inspired by AMES mutagenicity prediction. A coding agent converts
it to a two-site NVFLARE FedAvg job.

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
    ├── data/
    │   ├── train.csv       # synthetic training records
    │   ├── valid.csv       # synthetic validation records
    │   └── test.csv        # synthetic held-out records
    ├── model.py            # character-level PyTorch CNN
    ├── train.py            # standalone train and evaluation workflow
    └── requirements.txt    # source-project dependencies
```

## Data

The CSV files contain small, deterministic SMILES-like strings and synthetic
binary labels. They exercise molecular-string preprocessing without claiming
chemical validity or redistributing records from the AMES dataset. The source
uses a fixed character vocabulary so every future federated site assigns the
same token ID to the same character. Each row also has a stable `record_id` for
auditing generated site partitions. The classes deliberately overlap: neither
a single character pattern nor one structural motif determines the label. Each
split is balanced and includes 25% hard counterexamples to the dominant
motif-label association.

The converted simulation must deterministically partition the train,
validation, and test records into site-local files. It must not pool records
across sites.

## Model and Training

`source/model.py` defines `SmilesCNN`, a compact embedding and multi-kernel 1D
CNN. `source/train.py` uses PyTorch `Dataset` and `DataLoader` objects, a
persistent AdamW optimizer, weighted binary cross-entropy, and explicit train
and evaluation modes.

Standalone training runs for two epochs, selects a checkpoint by validation
AUROC, and reports loss, accuracy, and AUROC on held-out records. The conversion
must preserve the two-local-epoch training budget and convert training and
evaluation together. It should report validation AUROC to the federated
workflow for global model selection instead of reloading a site-specific best
checkpoint before model exchange.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
I have an existing PyTorch training project in ./source. Convert it to
federated learning using FedAvg and validate it locally with 2 clients and 2 rounds of training. You may download any required
public model artifacts, including tokenizer and configuration files, if they
are not already cached. Proceed without asking for additional confirmation.
```

## Run the Starting Project

Run the original standalone program with:

```bash
python source/train.py
```

The run writes `best_model.pt` and `metrics.json` under `source/outputs/`.
`MPLCONFIGDIR` defaults to the platform temporary directory (`/tmp` on Linux
and macOS) when it is not already set.

## Output Summary

The coding agent reports the simulation result and artifact paths. Review its
changes and results before using the pattern with real data.
