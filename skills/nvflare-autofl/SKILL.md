---
name: nvflare-autofl
description: "Optimize an existing NVFLARE job.py through an agent-assisted Auto-FL campaign that preserves FLARE execution, policy, artifacts, and reproducibility."
license: Apache-2.0
compatibility: "Requires NVFLARE 2.8.0+, Python, and permission to run NVFLARE jobs in the selected environment."
metadata:
  author: "nvflare"
  min_flare_version: "2.8.0"
  blast_radius: submits_production
  category: Optimization
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
boundaries, artifacts, and machine-readable contracts. The coding agent owns
hypotheses, source edits, new algorithm implementations, and candidate choice.

Resolve [run_job_campaign.py](scripts/run_job_campaign.py) relative to this
`SKILL.md`, store its absolute path as `RUNNER`, and initialize the campaign:

```bash
python "$RUNNER" initialize ./job.py [--metric <metric>] --mode <max|min> --env <sim|poc|prod> [--max-candidates <n>]
```

Read `autofl.yaml` and the JSON response, then prepare an agent-authored
candidate with a short hypothesis and optional candidate-only arguments:

```bash
python "$RUNNER" prepare ./job.py --name <candidate> --hypothesis "<expected improvement>" [--run-args "<args>"]
```

Edit only the returned candidate source directory. Modify existing allowed
files or add Python modules under the job root; do not edit the live best source
directly. Then evaluate the manifest:

```bash
python "$RUNNER" evaluate ./job.py --manifest <candidate_manifest.json>
```

Simulation evaluation runs the candidate immediately. POC and production
evaluation validates and materializes the candidate; submit it with standard
`nvflare job` commands, then call `record` with the manifest, job ID, artifacts,
and score. Use `abandon` to restore a pending candidate. Use `suggest` only when
you want deterministic tunable seeds; suggestions are never executed
automatically and do not limit agent-authored code candidates.

If the job directory contains a task-local `mutation_schema.yaml`, treat its
`comparison_budget_args.default_candidate_budget` and mutation bounds as
authoritative. Invalid generated proposals are product friction, not campaign
blockers; preserve the same campaign and continue with another same-budget
candidate.

Treat `job.py`, generated `autofl.yaml`, and optional job-local
`mutation_schema.yaml` as campaign inputs. Do not require example-specific
runbooks, branches, or initialization scripts.

Request escalated execution for the runner command because NVFLARE simulator
runs create local sockets that fail inside the restricted Codex sandbox. If a
runner command reports a sandbox/socket permission failure, treat it as an
infrastructure retry, not as a candidate result, and rerun the same command with
escalated execution.

The helper owns deterministic import, source snapshots, candidate validation,
execution, restoration, counting, ledger updates, campaign state, plotting,
and reports. After each lifecycle action, read
`.nvflare/autofl/campaign_state.json` and only finalize when
`final_response_allowed=true`. For long-running and simulator-stall handling, read
[continuous-campaigns.md](references/continuous-campaigns.md).
For comparison budgets, data distributions, and rerun evidence, read
[experiment comparability](references/experiment-comparability.md).

Read `autofl.yaml` and show the user a concise campaign summary:

- **Editable**: metric, environment, candidate budget, tunables, artifact
  locations, `objective.optimization_metric`, metric source, source hash, and importer version.
- **Unresolved**: dynamic defaults, unsupported Python semantics, missing
  metric sources, unknown data paths, or any low-confidence fields.
- **Allowed**: files the agent may edit, Python source it may create,
  fixed-budget fields it must preserve, and policy boundaries for the requested
  environment.

Treat `autofl.yaml` as the human-reviewable Auto-FL campaign config, not as a
replacement for `job.py` or an exported NVFLARE job folder. Use the original
`job.py` as the runnable experiment entry point throughout the candidate loop.

If `autofl.yaml` contains unresolved fields that affect execution safety,
candidate comparability, or production submission, ask the user to resolve those
specific fields before running candidates.

## Requirements
- Edit existing files only through candidate drafts and within
  `trust_contract.allowed_edit_paths`. New Python modules may match
  `trust_contract.allowed_create_patterns` under the job root.
- Use existing `mutation_schema.yaml` `preferred_targets` only after the runner
  reflects them in the trust contract; surface unresolved targets.
- You may create and register new Python server aggregators through `job.py`;
  do not limit exploration to existing FedAvg/FedAvgM/FedAdam/FedOpt/SCAFFOLD choices.
- Preserve `budget.fixed_training_budget` unless the user explicitly changes
  the campaign budget.
- If the environment provides `PYTHON`, `VIRTUAL_ENV`, or a venv on `PATH`,
  treat that prepared runtime as authoritative. Verify it, then use it for
  import, validation, execution, metric extraction, plotting, and reporting.
  Do not search for alternate interpreters or install dependencies unless the
  user explicitly asks you to prepare the environment.
- Treat generated `autofl.yaml`, task-local `mutation_schema.yaml`, and
  existing NVFLARE job/runtime configuration as authoritative. In the default
  simulation product flow, do not require task-local prose profiles, special
  branch setup, or harness initialization before invoking the runner.
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
1. Inspect `autofl.yaml`, current best source, prior manifests, and results.
2. Form a concrete hypothesis. Use literature, framework knowledge, source
   edits, new client or server algorithms, or a fallback tunable suggestion as
   appropriate.
3. Prepare a candidate, edit its draft, and evaluate its manifest.
4. Let the helper validate paths and fixed-budget comparability, compute the
   patch hash, execute or materialize it, extract metrics, and keep or restore.
5. Read campaign state and execute `next_action`. Run a source-backed literature
   pass when requested, then implement its strongest compatible idea.

## Continuous Campaign Rule
For uncapped campaigns, continue proposing and evaluating same-budget candidates
after setup and baseline until manually interrupted. Do not ask whether to keep
going or finalize while campaign state says `final_response_allowed=false`.

A kept improvement, refreshed plot, updated report, local commit, first plateau
check, or encoded `job.py` default is a checkpoint, not completion. Treat the
campaign state as authoritative: if `final_response_allowed=false`, execute
`next_action` and keep the same `job.py`, `autofl.yaml`, metric, environment,
ledger, and comparison budget. Use
[continuous-campaigns.md](references/continuous-campaigns.md) for simulator
watchdogs, campaign-state handling, and recovery rules.

## Candidate Caps

Campaigns are uncapped by default. If the user says "optimize this job" without
an explicit candidate budget, continue the campaign until manually interrupted
or blocked. Do not stop after the first baseline, first batch, first successful
candidate, first kept improvement, first local commit, or first plateau
checkpoint. Do not stop after a first sweep of tunables; broaden into
agent-authored code or literature-derived algorithm candidates. Uncapped progress
reports must not ask whether to continue; continue unless the
user explicitly interrupts or the code-owned state permits finalization.
Do not invent a replacement campaign or new objective after a recoverable
failure. Keep the current campaign identity and artifacts coherent unless the
human explicitly requests a new campaign.

If the user provides an `N`-candidate budget, pass it only through the runner's
explicit `--max-candidates` argument and count up to `N` comparable attempts
after baseline. Never infer a cap from an inherited environment variable. Do
not count import, validation, smoke runs, plotting, reporting, baseline, or
infrastructure-only retries. Count a real candidate crash after execution
starts. State must report `candidate_cap_source=explicit` or `uncapped`.

Treat plateau as a decision checkpoint, not an automatic stop. Summarize the
plateau in the running report, refresh `progress.png`, run the runner's `status`
action to refresh `.nvflare/autofl/campaign_state.json`, choose the returned
next mode, and continue unless the state reports `final_response_allowed=true`.
Use `campaign_guard.py` only for read-only ledger diagnostics; it never updates
authoritative campaign state.
After a source-backed review, record it with the helper's `record --literature
--hypothesis "<sources and decision>"` action before preparing its candidate.
After every literature-triggered plateau, evaluate at least one source-backed
server aggregation candidate, or record why the job contract is incompatible.

## Stop Handling

Only produce a final answer for a campaign when the code-owned campaign state
reports `final_response_allowed=true`, for example because the user manually
stopped it, an explicit cap is exhausted, production policy blocks execution, or
a hard safety/runtime blocker prevents further comparable runs. Then finalize
`results.tsv`, `progress.png`, and a concise report with baseline, best score,
metric source, failures, friction, commands, and absolute artifact paths.
