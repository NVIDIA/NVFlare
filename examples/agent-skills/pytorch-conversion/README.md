# Convert a Plain PyTorch Project

`source/` is a standalone, CPU-only PyTorch classifier. It deliberately has
no NVFLARE dependency.

## Prompt for Codex or Claude Code

```text
Use the installed nvflare-convert-pytorch skill to convert ./source into a
two-site NVFLARE FedAvg job. Preserve the model, cross-entropy evaluation, and
one local training epoch. Generate deterministic, site-local synthetic data
for simulation; do not pool records across sites. Create client.py and job.py
in ./source, validate the exported job with a local simulation, and summarize
the changed files and validation result.
```

## Review and run

The skill should leave the original `model.py` and `train.py` recognizable and
create `source/client.py` and `source/job.py`. Review those files, then use
the generated job's reported simulation command. A successful conversion must
return an evaluation metric and report the actual local optimizer-step count
for FedAvg weighting.

Install the listed dependencies before running the original source workload:

```bash
python -m pip install -r source/requirements.txt
python source/train.py
```
