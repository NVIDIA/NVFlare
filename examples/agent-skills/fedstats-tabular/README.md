# Federated Statistics for Tabular Data

Generate two site-local synthetic CSV files before asking the agent to analyze
them. No training code is needed for this workflow.

```bash
python generate_data.py
```

## Prompt for Codex or Claude Code

```text
Use the installed nvflare-fed-stats skill to compute federated count, mean,
standard deviation, and a 10-bin histogram for every numeric feature in
./data. Treat site-1 and site-2 as separate sites; do not combine raw rows.
Create client.py and job.py in this directory, run the generated job in local
simulation, and report only aggregate results and the output JSON path.
```

## Review and run

The generated job must retain the recipe's privacy cleaners and report
aggregates only. Review `client.py`, `job.py`, and the output JSON before
using the same workflow with sensitive data.

```bash
python -m pip install -r requirements.txt
```
