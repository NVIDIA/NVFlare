# Agent Skills: Convert a Hugging Face Trainer Project

This example demonstrates using the NVFLARE Convert Hugging Face Agent Skill
with Codex or Claude Code. The standalone Transformers `Trainer` project is
converted to a two-site NVFLARE FedAvg job.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r source/requirements.txt
npx skills add ../../../../skills --skill '*' -a codex -a claude-code -y
```

## Code Structure

```text
huggingface-conversion/
├── README.md
└── source/
    ├── train.py            # standalone Transformers Trainer workflow
    ├── train.jsonl         # synthetic training records
    ├── valid.jsonl         # synthetic validation records
    └── requirements.txt    # source-project dependencies
```

## Data

The JSONL files contain synthetic text classification records. The tiny public
model is downloaded on first run, so running the source workload requires
Internet access or a pre-populated Hugging Face cache. The conversion must
partition these records site-locally.

## Model

`source/train.py` constructs a Transformers `Trainer` with an accuracy metric
and a one-epoch local training budget.

## Run Agent Conversion

Open this directory in Codex or Claude Code and use this prompt:

```text
I have an existing Hugging Face Trainer project in ./source. Convert it to
federated learning and validate it locally.
```

## Run the Starting Project

Run the standalone source project with:

```bash
cd source
python train.py
```

## Output Summary

The coding agent reports the validation result and artifact paths. Review its
changes before using the pattern with real data.
