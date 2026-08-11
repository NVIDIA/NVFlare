# Convert a PyTorch Lightning Project

`source/` is a small standalone Lightning project with explicit training and
validation data loaders.

## Prompt for Codex or Claude Code

```text
Use the installed nvflare-convert-lightning skill to convert ./source into a
two-site NVFLARE FedAvg job. Preserve the LightningModule, validation metric,
and one-epoch local training budget. Use deterministic site-local synthetic
data for simulation, create client.py and job.py in ./source, validate the
exported job locally, and summarize the result.
```

## Review and run

Review the generated `source/client.py` and `source/job.py`, then use the
reported simulator command. The generated client should use the Lightning
Client API patch rather than replacing the `LightningModule` with a manual
PyTorch loop.

```bash
python -m pip install -r source/requirements.txt
python source/train.py
```
