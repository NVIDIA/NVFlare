# Federated Statistics for Image Data

Generate a small synthetic image dataset before asking the agent to analyze
it. The generator creates two site-local directories and no real images.

```bash
python -m pip install -r requirements.txt
python generate_data.py
```

## Prompt for Codex or Claude Code

```text
Use the installed nvflare-fed-stats skill to compute federated image count,
failure count, and a 10-bin pixel-intensity histogram across ./data. Treat
site-1 and site-2 as separate sites and do not move image files or pixels
between sites. Create client.py and job.py in this directory, run the generated
job in local simulation, and report aggregate results and the output JSON path.
```

## Review

One intentionally corrupt image is included so `failure_count` has observable
behavior. The generated job should use the image statistics path and report
only aggregate counts and histograms.
