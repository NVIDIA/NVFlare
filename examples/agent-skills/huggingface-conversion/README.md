# Convert a Hugging Face Trainer Project

`source/` is a standalone Transformers `Trainer` workload. The tiny public
model is downloaded on first run, so this example requires Internet access or
a pre-populated Hugging Face cache.

## Prompt for Codex or Claude Code

```text
Use the installed nvflare-convert-huggingface skill to convert ./source into a
two-site NVFLARE FedAvg job. Preserve the Trainer evaluation metric and
one-epoch local training budget. Keep all text records site-local when
partitioning the supplied JSONL data. Create client.py and job.py in ./source,
validate the exported job with a local simulation, and summarize the changed
files and validation result.
```

## Review and run

Review the generated client and job. The conversion should use
`flare.patch(trainer)` and retain the application-owned `Trainer` setup.

```bash
python -m pip install -r source/requirements.txt
cd source && python train.py
```
