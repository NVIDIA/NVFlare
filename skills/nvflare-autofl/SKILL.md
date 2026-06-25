---
name: nvflare-autofl
description: "Optimize an existing NVFLARE job.py through an agent-assisted Auto-FL campaign that preserves FLARE execution, policy, artifacts, and reproducibility."
min_flare_version: "2.8.0"
blast_radius: submits_production
skill_version: "0.1.0"
---

# NVFLARE Auto-FL

## Use When

Use this skill when the user asks to optimize an existing NVFLARE `job.py` for
accuracy, AUC, loss, runtime, robustness, or another metric in simulation, POC,
or production.

## Do Not Use When

Do not use for converting non-FL training code into NVFLARE, diagnosing failed
jobs without an optimization goal, production deployment setup, or generic
hyperparameter tuning outside an NVFLARE job.

## Workflow

Use this skill to optimize an existing NVFLARE `job.py` without asking the user
to learn a new Auto-FL command tree. The user selects this skill, points to a
job, and states the objective, environment, and optional budget. NVFLARE
provides the deterministic campaign import, execution substrate, policy
boundaries, artifacts, and machine-readable contracts. For simulation, the
coding agent invokes the product runner and monitors its code-owned state; the
runner owns candidate generation, execution, comparison, plots, and reports.
Manual agent edits are a fallback path, not the default product experience.

Before any candidate work, import the job deterministically:

```bash
python -m nvflare.app_common.autofl.job_importer ./job.py --metric <metric> --env <sim|poc|prod> --max-candidates <n> --output autofl.yaml
```

If the user did not specify a candidate cap, omit `--max-candidates` or leave it
unset in the review summary. Do not invent a default cap.

For simulation campaigns, use the bundled deterministic campaign runner as the
code-owned loop. Include
`--max-candidates <n>` only when the user gave an explicit candidate budget:

```bash
python "$CODEX_HOME/skills/nvflare-autofl/scripts/run_job_campaign.py" ./job.py --metric <metric> --env sim [--max-candidates <n>]
```

If the user did not specify a candidate cap, omit `--max-candidates`; the
runner will keep launching same-budget candidate attempts and refresh
`results.tsv`, `progress.png`, `.nvflare/autofl/campaign_state.json`, and
`autofl_report.md` after every finalized candidate until interrupted or blocked.
In this uncapped mode, do not ask the user whether to keep going. Report
checkpoint status as an observation, then continue monitoring or executing the
same runner while `final_response_allowed=false`.

If the job directory contains a task-local `mutation_schema.yaml`, treat its
comparison budget and mutation bounds as authoritative. Invalid generated
proposals are product friction, not campaign blockers; preserve the same
campaign and continue with another same-budget candidate.

Do not read or follow research harness prose, `program.md`, task profile
runbooks, `scripts/init_run.sh`, `.autoresearch` branch rules, or manual
research campaign instructions before starting the product runner. Those files
belong to the incubator workflow and can pull the agent back into an old
agent-driven loop. The product runner may read `mutation_schema.yaml`,
`autofl.yaml`, and the original `job.py`; use broader research instructions only
if the runner is unavailable or the user explicitly asks for the legacy research
campaign.

Request escalated execution for the runner command because NVFLARE simulator
runs create local sockets that fail inside the restricted Codex sandbox. If a
runner command reports a sandbox/socket permission failure, treat it as an
infrastructure retry, not as a candidate result, and rerun the same command with
escalated execution.

The runner owns deterministic import, baseline/candidate execution, candidate
counting, ledger updates, campaign state, progress plotting, and the concise
report. Do not produce a final response while the runner is active. After it
exits, read `.nvflare/autofl/campaign_state.json` and only finalize when
`final_response_allowed=true`. If an uncapped runner exits for a recoverable
runner/schema/simulator issue, repair the cause and resume the same requested
optimization once. For long-running and simulator-stall handling, read
[continuous-campaigns.md](references/continuous-campaigns.md).

Read `autofl.yaml` and show the user a concise campaign summary:

- **Editable**: metric, environment, candidate budget, tunables, artifact
  locations, source hash, and importer version.
- **Unresolved**: dynamic defaults, unsupported Python semantics, missing
  metric sources, unknown data paths, or any low-confidence fields.
- **Allowed**: files the agent may edit, fixed-budget fields it must preserve,
  and policy boundaries for the requested environment.

Treat `autofl.yaml` as the human-reviewable Auto-FL campaign config, not as a
replacement for `job.py` or an exported NVFLARE job folder. Use the original
`job.py` as the runnable experiment entry point throughout the candidate loop.

If `autofl.yaml` contains unresolved fields that affect execution safety,
candidate comparability, or production submission, ask the user to resolve those
specific fields before running candidates.

## Requirements

- Do not edit outside `job.allowed_edit_paths`.
- Preserve `budget.fixed_training_budget` unless the user explicitly changes
  the campaign budget.
- If the environment provides `PYTHON`, `VIRTUAL_ENV`, or a venv on `PATH`,
  treat that prepared runtime as authoritative. Verify it, then use it for
  import, validation, execution, metric extraction, plotting, and reporting.
  Do not search for alternate interpreters or install dependencies unless the
  user explicitly asks you to prepare the environment.
- Treat generated `autofl.yaml`, task-local `mutation_schema.yaml`, and
  existing NVFLARE job/runtime configuration as authoritative. In the default
  simulation product flow, do not require task-local prose profiles, campaign
  branch setup, or research harness initialization before invoking the runner.
- Use NVFLARE's existing execution surfaces:
  - For simulation, run the imported job with its configured `SimEnv`.
  - For POC and production, use standard `nvflare job submit`, `job wait`,
    `job download`, and related job/status commands.
- Record every candidate in a ledger such as `results.tsv` with a short name,
  changed files, diff summary, run command, metric result, artifacts, and
  failure reason when applicable.
- Prefer small, reviewable edits over broad rewrites.
- Treat production as an available execution environment, but never bypass
  startup-kit authentication, site policy, or normal NVFLARE job submission.

## Candidate Loop

For simulation (`--env sim`), prefer `scripts/run_job_campaign.py` for both
explicitly capped and uncapped campaigns. Start the runner before inspecting or
editing task code beyond the deterministic import and `mutation_schema.yaml`
review. Use the manual loop below only when the runner is unavailable, when the
requested environment is POC/production, or when the user explicitly asks for
source-code mutations that the runner cannot express yet.

1. Inspect `autofl.yaml`, the allowed files, and the current job behavior.
2. Propose and run a candidate tied to supported tunables or allowed files.
3. Validate importability and fixed-budget comparability.
4. Extract the requested metric from NVFLARE artifacts/logs.
5. Update `results.tsv`, mark non-survivors as `discard`, crashes as `crash`,
   the survivor as `keep`, and unresolved active rows as `candidate`.
6. Refresh `progress.png`, update campaign state, and launch the next
   comparable candidate batch unless the code-owned state says
    `final_response_allowed=true` or production policy blocks execution.

## Autoresearch Operating Rule

For uncapped campaigns, behave like the original Auto-FL research loop: after
setup and baseline, continue launching same-budget candidate batches until
manually interrupted. Do not ask whether to keep going. Do not produce a final
answer from your own judgment while the code-owned campaign state says
`final_response_allowed=false`.

A kept improvement, refreshed plot, updated report, local commit, first plateau
check, or encoded `job.py` default is a checkpoint, not completion. Treat the
campaign state as authoritative: if `final_response_allowed=false`, execute
`next_action` and keep the same `job.py`, `autofl.yaml`, metric, environment,
ledger, and comparison budget. Use
[continuous-campaigns.md](references/continuous-campaigns.md) for simulator
watchdogs, legacy `.autoresearch` guard handling, and recovery rules.

## Candidate Caps

Campaigns are uncapped by default. If the user says "optimize this job" without
an explicit candidate budget, continue the campaign until manually interrupted
or blocked. Do not stop after the first baseline, first batch, first successful
candidate, first kept improvement, first local commit, or first plateau
checkpoint. Do not stop after a first sweep of available CLI tunables; in
uncapped mode repeating, broadening, or generated same-budget candidates are
valid continued campaign work. Progress reports in uncapped mode must not be
phrased as "should I continue?" decisions; the answer is continue unless the
user explicitly interrupts or the code-owned state permits finalization.
Do not invent a replacement campaign or new objective after a recoverable
failure. Keep the current campaign identity and artifacts coherent unless the
human explicitly requests a new campaign.

If the user provides an `N`-candidate budget, count up to `N` comparable
candidate attempts after the baseline. Do not count deterministic import,
validation, smoke runs, plotting, reporting, the baseline, or
infrastructure-only retries caused by sandbox/socket/runtime setup. Count a real
candidate crash once the candidate run starts under the intended execution
environment.

Treat plateau as a decision checkpoint, not an automatic stop. Summarize the
plateau in the running report, refresh `progress.png`, run the campaign guard or
read the runner's `.nvflare/autofl/campaign_state.json`, choose the returned
next mode, and continue unless the state reports `final_response_allowed=true`.

## Stop Handling

Only produce a final answer for a campaign when the code-owned campaign state
reports `final_response_allowed=true`, for example because the user manually
stopped it, an explicit cap is exhausted, production policy blocks execution, or
a hard safety/runtime blocker prevents further comparable runs. At that point,
end with finalized `results.tsv`, refreshed `progress.png`, a concise report
covering baseline, best score, artifacts, failures, product friction, and
reproduction commands, plus absolute paths to all final artifacts.
