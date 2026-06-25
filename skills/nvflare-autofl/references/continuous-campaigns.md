# Continuous Campaigns

Use this reference when an Auto-FL campaign is uncapped, long-running, or
recovering from a simulator stall. The top-level skill owns the interaction
contract; this file carries the operational detail that keeps the campaign from
prematurely stopping.

## Runner State

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

During long simulations, monitor the active process plus the current
`autofl_runs/<candidate>/run.log`. A live process with no final ledger row is a
running candidate, not a reason to stop. If logs are temporarily quiet but CPU or
GPU use and the child process remain active, keep waiting.

## Campaign Guards

If `.autoresearch/campaign_state.json` exists, read it before any final
response. If `scripts/campaign_guard.py` exists, run it after every checkpoint
and before any final response. If the state has `final_response_allowed=false`,
execute `next_action` immediately; the skill text is only the interaction layer.

Common next actions:

- `finalize_pending_candidates`: finalize reviewed candidate rows and rerun the
  guard.
- `run_literature_loop`: run the literature loop and launch source-backed
  candidates.
- `launch_next_candidate_batch`: choose a safe same-budget axis and launch the
  next batch.

After every finalized batch, run the available plateau or progress watchdog when
the task provides one. If it recommends `continue`, refresh `progress.png` and
keep iterating locally. If it recommends a literature or exploration mode, record
that decision in the ledger/report, refresh `progress.png`, and launch the top
compatible candidate batch next. If no non-duplicate safe local axis remains,
switch mode rather than stopping: broaden the search within `autofl.yaml`, run a
literature-inspired proposal pass, revisit unresolved-but-safe tunables, or let
the deterministic uncapped runner continue its generated candidate stream.

## Simulator Recovery

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
comparison budget, then continue the same campaign.

The product runner includes these simulator-stall watchdogs; prefer letting it
record the crash row, refresh artifacts, and launch the next candidate. For
legitimately long quiet tasks, raise `--simulator-no-progress-timeout`, set
`AUTOFL_SIMULATOR_NO_PROGRESS_TIMEOUT_SECONDS`, or set
`simulator_no_progress_timeout_seconds` in the task profile.

If the current process has already stopped but the user did not ask to stop, do
not leave the campaign in a terminal state. Inspect the campaign state and
ledger, fix the recoverable cause, and continue the same optimization with the
same `job.py`, `autofl.yaml`, metric, environment, and comparison budget.
