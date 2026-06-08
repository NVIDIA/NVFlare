---
name: nvflare-autofl
description: Optimize an existing NVFlare job through an agent-assisted Auto-FL workflow. Use when a user asks to improve accuracy, AUC, loss, runtime, or robustness of an NVFlare job in simulation, POC, or production while preserving NVFlare execution, policy, artifacts, and reproducibility.
---

# NVFlare Auto-FL

Use this skill to optimize an existing NVFlare `job.py` without asking the user
to learn a new Auto-FL command tree. The user selects this skill, points to a
job, and states the objective, environment, and budget. NVFlare provides the
deterministic import, execution substrate, policy boundaries, artifacts, and
machine-readable contracts. The coding agent plans, edits allowed files, runs
candidates, compares results, and reports evidence.

## Required First Step

Before editing files, import the job deterministically:

```bash
python -m nvflare.app_common.autofl.job_importer ./job.py --metric <metric> --env <sim|poc|prod> --max-candidates <n> --output autofl.yaml
```

Read `autofl.yaml` and show the user a concise trust summary:

- **Extracted**: recipe or FedJob surface, train script, metric, environment,
  fixed budget, tunables, artifact locations, and provenance.
- **Unresolved**: dynamic defaults, unsupported Python semantics, missing
  metric sources, unknown data paths, or any low-confidence fields.
- **Allowed**: files the agent may edit, fixed-budget fields it must preserve,
  and policy boundaries for the requested environment.

If `autofl.yaml` contains unresolved fields that affect execution safety,
candidate comparability, or production submission, ask the user to resolve those
specific fields before running candidates.

## Operating Rules

- Do not edit outside `job.allowed_edit_paths`.
- Preserve `budget.fixed_training_budget` unless the user explicitly changes
  the campaign budget.
- Use NVFlare's existing execution surfaces:
  - For simulation, run the imported job with its configured `SimEnv`.
  - For POC and production, use standard `nvflare job submit`, `job wait`,
    `job download`, and related job/status commands.
- Record every candidate with a short name, changed files, diff summary,
  run command, metric result, artifacts, and failure reason when applicable.
- Prefer small, reviewable edits over broad rewrites.
- Treat production as an available execution environment, but never bypass
  startup-kit authentication, site policy, or normal NVFlare job submission.

## Candidate Loop

1. Inspect `autofl.yaml`, the allowed files, and the current job behavior.
2. Propose a candidate change tied to one or more supported tunables or files.
3. Edit only allowed files.
4. Validate the job can still be imported and the fixed budget still matches.
5. Run the candidate through NVFlare in the requested environment.
6. Extract the requested metric from NVFlare artifacts/logs.
7. Update the candidate ledger and keep the best reproducible candidate.
8. Stop when the budget is exhausted, the user stops the run, or results plateau.

## Final Report

End with:

- Best candidate and metric improvement.
- Baseline versus candidate leaderboard.
- Files changed and why.
- Artifacts and commands needed to reproduce the best result.
- Unresolved limitations or production review items.
