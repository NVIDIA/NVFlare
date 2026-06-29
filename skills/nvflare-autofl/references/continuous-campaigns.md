# Continuous Campaigns

Use this reference when an Auto-FL campaign is uncapped, long-running, or
recovering from a simulator stall. The top-level skill owns the interaction
contract; this file carries the operational detail that keeps the campaign from
prematurely stopping.

## Lifecycle State

Each `scripts/run_job_campaign.py` lifecycle action exits with a JSON envelope;
the campaign continues through `.nvflare/autofl/campaign_state.json`. In
uncapped mode, a completed action, current best, plot, report, or exhausted local
tunable sweep is a checkpoint only. Execute `next_action` while
`final_response_allowed=false`.

A prepared manifest is pending work. Edit its candidate source and evaluate it,
or abandon it explicitly; do not silently start another candidate. Invalid
drafts are product friction to repair and reevaluate, not a reason to terminate
the campaign.

During a long `evaluate` action, monitor the process plus
`autofl_runs/<candidate>/run.log`. A live process with no final ledger row is a
running candidate. If logs are temporarily quiet but CPU or GPU use and the
child process remain active, keep waiting.

## Campaign Guards

The product runner writes `.nvflare/autofl/campaign_state.json` through
`scripts/campaign_guard.py`; read that state before any final response. This
product state is authoritative. If it has `final_response_allowed=false`,
execute `next_action` immediately; the skill text is only the interaction layer.

Common next actions:

- `edit_candidate` or `evaluate_candidate`: finish the pending candidate draft.
- `propose_candidate`: form a hypothesis, prepare its manifest, and edit the
  returned candidate source directory.
- `submit_baseline` or `submit_candidate`: use the standard POC/production job
  lifecycle, then call `record` with its job ID and artifacts.
- `run_literature_loop`: run a short source-backed literature pass, record a
  non-scored `literature` row when a ledger is available, then launch the next
  compatible same-budget candidates.

After every finalized batch, run the available plateau or progress watchdog when
the task provides one. If it recommends `continue`, refresh `progress.png` and
keep iterating locally. If it recommends a literature or exploration mode, record
that decision in the ledger/report, refresh `progress.png`, and launch the top
compatible candidate batch next. If no non-duplicate safe local axis remains,
switch mode rather than stopping: broaden the search within `autofl.yaml`, run a
literature-inspired proposal pass, implement a compatible algorithm change, or
request deterministic tunable suggestions as seeds.

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
