# Agent Skills: Convert a Hugging Face Trainer Project

This example demonstrates using the NVFLARE Convert Hugging Face Agent Skill
with Codex or Claude Code. The standalone Transformers `Trainer` project is
converted to a two-site NVFLARE FedAvg job.

## NVIDIA FLARE Installation

For complete setup instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
python -m pip install 'nvflare~=2.9.0'
python -m pip install -r source/requirements.txt
npx skills add ../../../../skills -a codex -a claude-code
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

The coding agent adds `source/client.py` and `source/job.py` during the
conversion.

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
Use the installed nvflare-convert-huggingface skill to convert ./source into a
two-site NVFLARE FedAvg job. Preserve the Trainer evaluation metric and
one-epoch local training budget. Keep all text records site-local when
partitioning the supplied JSONL data. Create client.py and job.py in ./source,
validate the exported job with a local simulation, and summarize the changed
files and validation result.
```

## Run Job

Run the standalone source project with:

```bash
cd source
python train.py
```

After reviewing the generated client and job, run the federated simulation:

```bash
python job.py
```

## Output Summary

A successful conversion retains the application-owned `Trainer` setup and
uses `flare.patch(trainer)` for federated exchange. The coding agent reports
the validation result and generated-job artifact paths.
