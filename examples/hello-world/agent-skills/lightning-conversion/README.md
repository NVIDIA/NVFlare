# Agent Skills: Convert a PyTorch Lightning Project

This example demonstrates using the NVFLARE Convert Lightning Agent Skill with
Codex or Claude Code. The starting project is a standalone, CPU-first Lightning
classifier inspired by AMES mutagenicity prediction. A coding agent converts
it to a two-site NVFLARE FedAvg job.

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
    ├── data/
    │   ├── train.csv       # synthetic training records
    │   ├── valid.csv       # synthetic validation records
    │   └── test.csv        # synthetic held-out records
    ├── model.py            # Lightning character-level CNN and metrics
    ├── train.py            # standalone Trainer workflow
    └── requirements.txt    # source-project dependencies
```

## Data

The CSV files contain small, deterministic SMILES-like strings and synthetic
binary labels. They exercise molecular-string preprocessing without claiming
chemical validity or redistributing records from the AMES dataset. A fixed
character vocabulary keeps token IDs consistent across future federated sites,
and stable `record_id` values support partition auditing.

Every split is balanced and includes 25% hard counterexamples to the dominant
motif-label association. The converted simulation must deterministically
partition the train, validation, and test records into site-local files. It
must not pool records across sites.

## Model and Training

`source/model.py` defines `LitSmilesCNN`, including its AdamW optimizer,
training step, validation and test steps, and epoch-level accuracy and AUROC
metrics through `self.log`.

Standalone training runs for two epochs with a native Lightning `Trainer` and
selects `best_model.ckpt` by `val_auroc`. The conversion must preserve the
two-local-epoch budget and Lightning-native evaluation. It should report
`val_auroc` to the federated workflow for global model selection instead of
reloading a site-specific best checkpoint before model exchange.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
I have an existing PyTorch Lightning training project in ./source. Convert it
to federated learning and validate it locally. You may download any required
public model artifacts, including tokenizer and configuration files, if they
are not already cached. Proceed without asking for additional confirmation.
```

## Run the Starting Project

Run the starting standalone project with:

```bash
python source/train.py
```

The run writes `best_model.ckpt` and `metrics.json` under `source/outputs/`.
Expected Lightning setup messages and harmless small-loader warnings are
suppressed so the console shows one metrics line per epoch and a short final
summary. `MPLCONFIGDIR` defaults to the platform temporary directory (`/tmp` on
Linux and macOS) when it is not already set. Errors remain visible.

## Output Summary

The coding agent reports the validation result and artifact paths. Review its
changes before using the pattern with real data.
