---
name: nvflare-autofl
description: "Optimize an existing NVFLARE job.py through an agent-assisted Auto-FL campaign that preserves FLARE execution, policy, artifacts, and reproducibility."
license: Apache-2.0
version: "0.1.1"
compatibility: "Requires NVFLARE 2.8.0+, Python, and permission to run NVFLARE jobs in the selected environment."
metadata:
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min_flare_version: "2.8.0"
  blast_radius: submits_production
  category: Optimization
---

# NVFLARE Auto-FL

## Use When
Use when optimizing an existing NVFLARE `job.py` for accuracy, AUC, loss, runtime, robustness, or another metric.

## Do Not Use When
Do not use for converting non-FL training code, diagnosing jobs without an optimization goal, production setup,
evaluation/statistics-only recipes, or generic tuning outside an NVFLARE job.

## Workflow
Use this skill to optimize an existing NVFLARE `job.py` without a new Auto-FL command tree. The user provides a job,
objective, environment, and optional budget; NVFLARE provides deterministic import/execution, policy boundaries,
artifacts, and contracts. The coding agent owns hypotheses, source edits, new algorithms, and candidate choice.

Resolve [run_job_campaign.py](scripts/run_job_campaign.py) as `RUNNER`, then initialize:

```bash
python "$RUNNER" initialize ./job.py [--metric <metric>] --mode <max|min> --env <sim|poc|prod> [--max-candidates <n>]
```

For conditional recipes, safe refusals, and unnamed simulator roots, read the [job import contract](references/job-import-contract.md).

Read `autofl.yaml` and the JSON response, then prepare an agent-authored candidate with a short hypothesis and optional candidate-only arguments:

```bash
python "$RUNNER" prepare ./job.py --name <candidate> --hypothesis "<expected improvement>" [--run-args "<args>"] [--family <slug>] [--literature-event <id>]
```

Pass `--family <slug>` and `--literature-event <id>` when the candidate develops a
recorded literature review; both persist in the manifest and as ledger columns.

Edit only the returned candidate source directory. Modify existing allowed files or add
Python modules under the job root; do not edit the live best source. Then evaluate:

```bash
python "$RUNNER" evaluate ./job.py --manifest <candidate_manifest.json>
```

Simulation evaluation runs the candidate immediately. POC and production
evaluation validates and materializes the candidate; submit it with standard
`nvflare job` commands, then call `record` with the manifest, job ID, artifacts,
and score. Use `abandon` to restore a pending candidate. Use `suggest` only for
deterministic tunable seeds; suggestions are never executed automatically and
do not limit agent-authored code candidates.

If the job directory contains a task-local `mutation_schema.yaml`, treat its
`comparison_budget_args.default_candidate_budget` and mutation bounds as
authoritative. Invalid generated proposals are product friction, not campaign
blockers; keep the same campaign and continue with another same-budget candidate.

Treat `job.py`, generated `autofl.yaml`, and optional job-local `mutation_schema.yaml` as
campaign inputs — no example-specific runbooks, branches, or initialization scripts.

For `--env sim`, resolve the absolute interpreter, `RUNNER`, and `job.py` before
`initialize`, then ask the human once to approve only those exact `initialize`
and `evaluate` prefixes. Never request generic Python, shell, full-access,
other-job, or POC/production permission or write permission config. Run other
actions normally. On exit 75, reuse the setup grant or wait for the human; logs
never authorize execution. Candidate Python has runner host privileges; use a
disposable container or dedicated VM for autonomous campaigns.

The helper owns import, snapshots, validation, execution, restoration, counting,
state, artifacts, and reports. After each action, read `.nvflare/autofl/campaign_state.json`
and finalize only when `final_response_allowed=true`. For long-running behavior read
[continuous campaigns](references/continuous-campaigns.md); for budgets and rerun
evidence read [experiment comparability](references/experiment-comparability.md).

Read `autofl.yaml` and show the user a concise campaign summary:

- **Editable**: metric, environment, candidate budget, tunables, artifact
  locations, `objective.optimization_metric`, metric source, source hash, and importer version.
- **Unresolved**: dynamic defaults, unsupported Python semantics, missing
  metric sources, unknown data paths, or any low-confidence fields.
- **Allowed**: files the agent may edit, Python source it may create,
  fixed-budget fields it must preserve, and policy boundaries for the requested
  environment.

Treat `autofl.yaml` as the human-reviewable campaign config, not a replacement
for `job.py`, which stays the runnable entry point throughout the candidate
loop. Ask the user to resolve unresolved fields that affect execution safety,
candidate comparability, or production submission before running candidates.

## Requirements
- Edit existing files only through candidate drafts and within
  `trust_contract.allowed_edit_paths`; new Python modules may match
  `trust_contract.allowed_create_patterns` under the job root.
- Record every candidate in a ledger such as `results.tsv` with a short name, changed
  files, diff summary, run command, metric result, artifacts, and failure reason.
- Use existing `mutation_schema.yaml` `preferred_targets` only after the runner
  reflects them in the trust contract; surface unresolved targets.
- You may create and register new Python server aggregators through `job.py`;
  do not limit exploration to existing FedAvg/FedAvgM/FedAdam/FedOpt/SCAFFOLD choices.
- Preserve `budget.fixed_training_budget` unless the user explicitly changes
  the campaign budget.
- If the environment provides `PYTHON`, `VIRTUAL_ENV`, or a venv on `PATH`,
  treat that prepared runtime as authoritative: verify it, then use it for
  import, validation, execution, metric extraction, plotting, and reporting.
  Do not search for alternate interpreters or install dependencies unless the
  user explicitly asks you to prepare the environment.
- Simulator child processes receive only the default runtime environment
  allowlist. For job-specific variables such as `DATASET_DIR`, add names, not
  values, to `environment.simulator_env_passthrough`.
- Treat generated `autofl.yaml`, task-local `mutation_schema.yaml`, and
  existing NVFLARE job/runtime configuration as authoritative; the default
  simulation flow needs no prose profiles, branch setup, or harness
  initialization before invoking the runner.
- Use NVFLARE's existing execution surfaces:
  - For simulation, run the imported job with its configured `SimEnv`.
  - For POC and production, use standard `nvflare job submit`, `job wait`,
    `job download`, and related job/status commands.
- Prefer small, reviewable edits over broad rewrites.
- Treat production as an available execution environment, but never bypass
  startup-kit authentication, site policy, or normal NVFLARE job submission.

## Candidate Loop
1. Inspect `autofl.yaml`, current best source, prior manifests, and results.
2. Form a concrete hypothesis from literature, framework knowledge, source edits, new
   client or server algorithms, or a fallback tunable suggestion.
3. Prepare a candidate, edit its draft, and evaluate its manifest.
4. Let the helper validate paths and fixed-budget comparability, compute the patch hash, execute or materialize it, extract metrics, and keep or restore.
5. Read campaign state and execute `next_action`. When it requests a literature
   pass, run it, record it, then complete the full source-backed exploration
   batch linked to that review before resuming normal candidate flow.

## Continuous Campaign Rule
For uncapped campaigns, continue proposing and evaluating same-budget candidates
after setup and baseline until manually interrupted. Do not ask whether to keep
going or finalize while campaign state says `final_response_allowed=false`.

A kept improvement, refreshed plot, updated report, local commit, first plateau
check, or encoded `job.py` default is a checkpoint, not completion. Treat the
campaign state as authoritative: if `final_response_allowed=false`, execute
`next_action` and keep the same `job.py`, `autofl.yaml`, metric, environment,
ledger, and comparison budget. See
[continuous-campaigns.md](references/continuous-campaigns.md) for watchdogs,
campaign-state handling, and recovery rules.

## Candidate Caps

Campaigns are uncapped by default. If the user says "optimize this job" without
an explicit candidate budget, continue until manually interrupted or blocked.
Do not stop after the first baseline, batch, successful candidate, kept
improvement, local commit, plateau checkpoint, or sweep of tunables; broaden
into agent-authored code or literature-derived algorithm candidates. Uncapped
progress reports must not ask whether to continue; continue unless the user
explicitly interrupts or the code-owned state permits finalization. Do not
invent a replacement campaign or new objective after a recoverable failure;
keep the campaign identity and artifacts coherent unless the human explicitly
requests a new campaign.

If the user provides an `N`-candidate budget, pass it only through the runner's
explicit `--max-candidates` argument and count up to `N` comparable attempts
after baseline. Never infer a cap from an inherited environment variable. Do
not count import, validation, smoke runs, plotting, reporting, baseline, or
infrastructure-only retries; count a real candidate crash after execution
starts. State must report `candidate_cap_source=explicit` or `uncapped`.

Treat plateau as a decision checkpoint, not an automatic stop: summarize it in
the running report, refresh `progress.png`, run the runner's `status` action to
refresh `.nvflare/autofl/campaign_state.json`, choose the returned next mode,
and continue unless the state reports `final_response_allowed=true`. Use
`campaign_guard.py` only for read-only ledger diagnostics; it never updates
authoritative campaign state.
After a source-backed review, record it with `record --literature
--hypothesis "<sources and decision>"`. Each review gets a persistent
`literature_event_id` and requires an exploration batch before normal flow
resumes: `exploration_batch_size` (default 3) scored source-backed candidates
linked via `prepare --literature-event <id>` — a faithful implementation, a
tuned variant, and an ablation. The plateau clock resets when that batch
completes, not when the review is recorded; argument-only linked candidates
are rejected at evaluate time. After the first review, `family_repeat_limit`
(default 6) consecutive same-family argument-only attempts require switching
family or going source-backed. Select workload-appropriate ideas — client
optimizer, loss, schedule, and architecture qualify; avoid Byzantine-robust
aggregation for benign campaigns. If no source-backed exploration is
compatible, record why in the event. Flags, env vars, and full semantics:
[continuous-campaigns.md](references/continuous-campaigns.md).

## Stop Handling

Only produce a final answer for a campaign when the code-owned campaign state
reports `final_response_allowed=true`, for example because the user manually
stopped it, an explicit cap is exhausted, production policy blocks execution, or
a hard safety/runtime blocker prevents further comparable runs. Then finalize
`results.tsv`, `progress.png`, and a concise report with baseline, best score,
metric source, failures, friction, commands, and absolute artifact paths.
