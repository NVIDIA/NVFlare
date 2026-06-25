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
comparison budget and mutation bounds as authoritative. The runner reads it
directly when present, so realistic tasks can supply fixed data/round/evaluation
contracts without requiring a long user prompt.
Candidates outside `mutation_schema.yaml` bounds are invalid proposals, not
campaign blockers. The runner must skip them before execution when possible and
continue with another same-budget candidate. The agent must not stop the
campaign, finalize a report, or start a new campaign merely because an invalid
generated proposal was skipped or crashed.

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
`final_response_allowed=true`. If an uncapped runner exits because of an invalid
generated candidate or other recoverable runner bug, fix the runner or schema
locally, then resume or relaunch the same requested optimization once. Do not
convert that into a completed campaign and do not start a different campaign.
During long simulations, monitor the active process plus the current
`autofl_runs/<candidate>/run.log`. A live process with no final ledger row is a
running candidate, not a reason to stop. If logs are temporarily quiet but CPU or
GPU use and the child process remain active, keep waiting.
For NVFLARE simulator runs, the server log can be quiet after it dispatches a
round while individual clients are still training. Before declaring a stall,
inspect the active simulator directory under `/tmp/nvflare/simulation/<run>` and
check `site-*/log.txt` or `site-*/log_fl.txt` for epoch, finished-training,
download, or task-completion progress. If any client log or server aggregation
marker advances within the expected candidate runtime, continue the same
candidate; do not stop the runner, final-answer, or start a new campaign.
If the active NVFLARE simulator logs show a hard child-process connection
failure such as
`Failed to create connection to the child process in SimulatorClientRunner`, or
if a dispatched simulator round has no advancing server/client progress markers
past the configured no-progress watchdog timeout, do not start a new campaign and
do not produce a final report. Treat only that active candidate as crashed,
terminate the stuck `job.py` child if the runner has not already done so,
preserve the same `job.py`, `autofl.yaml`, metric, environment, ledger, and
comparison budget, then continue the same campaign. The product runner includes
these simulator-stall watchdogs; prefer letting it record the crash row, refresh
artifacts, and launch the next candidate. For legitimately long quiet tasks,
raise `--simulator-no-progress-timeout`, set
`AUTOFL_SIMULATOR_NO_PROGRESS_TIMEOUT_SECONDS`, or set
`simulator_no_progress_timeout_seconds` in the task profile.

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
2. Propose a candidate change tied to one or more supported tunables or files.
3. Edit only allowed files.
4. Validate the job can still be imported and the fixed budget still matches.
5. Run the candidate through NVFLARE in the requested environment with a unique
   run name, log path, and artifact path.
6. Extract the requested metric from NVFLARE artifacts/logs.
7. Update the candidate ledger. Mark reviewed non-survivors as `discard`,
   crashes as `crash`, the survivor as `keep`, and leave only unresolved active
   rows as `candidate`.
8. Commit or checkpoint the ledger and surviving code change when a candidate is
   kept, then refresh the code-owned campaign state before choosing the next
   mutation axis.
9. Refresh `progress.png` from the ledger after every finalized batch, plateau
   checkpoint, cap exhaustion, manual stop, or hard-blocker checkpoint.
10. Run the task-local campaign guard when present, for example
    `scripts/campaign_guard.py results.tsv --state .autoresearch/campaign_state.json --format json`.
    If no task-local guard exists, read the runner's
    `.nvflare/autofl/campaign_state.json` state file.
11. Report the mutation hypothesis, changed files, commands run, observed
    outcome, literature basis when relevant, run analysis, and next mutation.
12. Launch the next comparable candidate batch unless the code-owned state says
    `final_response_allowed=true` or production policy blocks execution.

## Autoresearch Operating Rule

For uncapped campaigns, behave like the original Auto-FL research loop: after
setup and baseline, continue launching same-budget candidate batches until
manually interrupted. Do not ask whether to keep going. Do not produce a final
answer from your own judgment while the code-owned campaign state says
`final_response_allowed=false`.

When `scripts/run_job_campaign.py` is running without `--max-candidates`, never
send Ctrl-C, interrupt the background terminal, or stop the runner because a
first sweep is complete, duplicate-cycle candidates started, a current best
looks clear, a plot/report exists, or no new obvious local axis remains. In
uncapped mode those are monitoring observations only. The user owns manual
interruption; the agent may inspect the ledger/state and report progress, but
must leave the runner active while `final_response_allowed=false`.
Also never interrupt an uncapped runner because it attempted, skipped, or
recorded an invalid candidate proposal. Invalid proposals should be filtered by
the runner and treated as product friction to repair while preserving the
long-running optimization intent.

A kept improvement, refreshed plot, updated report, local commit, or encoded
`job.py` default is a checkpoint, not completion. After encoding or validating a
best-so-far default, run the watchdog, update the ledger/report, and launch the
next same-budget candidate batch. Do not final-answer after the first baseline,
first successful candidate, first improvement, first local commit, first report
update, first `progress.png`, first plateau check, or first default update.

Treat the campaign state as authoritative:

- If `.autoresearch/campaign_state.json` exists, read it before any final
  response.
- If `scripts/campaign_guard.py` exists, run it after every checkpoint and
  before any final response.
- If the state has `final_response_allowed=false`, execute `next_action`
  immediately; the skill text is only the interaction layer.
- If the state has `next_action=finalize_pending_candidates`, finalize reviewed
  candidate rows and rerun the guard.
- If the state has `next_action=run_literature_loop`, run the literature loop
  and launch source-backed candidates.
- If the state has `next_action=launch_next_candidate_batch`, choose a safe
  same-budget axis and launch the next batch.

After every finalized batch, run the available plateau or progress watchdog when
the task provides one. If it recommends `continue`, refresh `progress.png` and
keep iterating locally. If it recommends a literature or exploration mode, record
that decision in the ledger/report, refresh `progress.png`, and launch the top
compatible candidate batch next. If no non-duplicate safe local axis remains,
switch mode rather than stopping: broaden the search within `autofl.yaml`, run a
literature-inspired proposal pass, revisit unresolved-but-safe tunables, or let
the deterministic uncapped runner continue its generated candidate stream. Stop
only when the user manually interrupts, an explicit candidate cap is exhausted,
production policy blocks execution, or repeated failures share a hard root cause
that prevents safe comparable runs.
If the current process has already stopped but the user did not ask to stop, do
not leave the campaign in a terminal state. Inspect the campaign state and
ledger, fix the recoverable cause, and continue the same optimization with the
same `job.py`, `autofl.yaml`, metric, environment, and comparison budget.
If a runner or candidate process stalls on a known NVFLARE simulator child
connection timeout or a configured simulator no-progress watchdog, recover the
active candidate in place. Do not kick off a new campaign directory, new prompt,
new objective, or new baseline unless the human explicitly requests that reset.

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
end the search with:

- finalized `results.tsv` or equivalent candidate ledger;
- refreshed `progress.png` or an explicit explanation if plotting is impossible;
- concise report or run summary with baseline, best score, leaderboard, files
  changed, artifacts, command provenance, failures, product friction, and
  reproduction commands;
- absolute paths to `autofl.yaml`, `results.tsv`, `progress.png`, campaign state,
  and any report artifact in the final answer.
