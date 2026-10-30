# Resource statistics: runtime integration

This page describes the components and lifecycle of resource statistics introduced
in the [overview](resource_statistics_design.md). The [measurement design](resource_statistics_measurement.md)
defines the algorithms; the [reporting design](resource_statistics_reporting.md)
defines records, validation, delivery and queries. The internal interfaces below
keep resource accounting separate from runtime control and application code.

## Components and ownership

The internal `nvflare/private/fed/resource_stats/` package owns collection and
reporting. Its interfaces are integration points within the runtime, not extension
APIs for job code.

| Component | Input and result | Responsibility |
| --- | --- | --- |
| `contract.py` | Untrusted JSON/measurements to normalized records; participant entries to job/study totals | Validates versions, shapes, numbers and sizes, performs decimal arithmetic, and propagates quality. The contract owns neither runtime nor filesystem operations. |
| `accumulator.py:ResourceCapacityRecorder` | One probe snapshot to normalized startup capacity | Preserves independent dimensions and consolidates equivalent hardware groups without a separate timer. |
| `collector.py:ExecutionResourceCollector` | Finalized execution environment to one immutable `stop()` snapshot | Probes once, opens the shared CPU and capacity window, and freezes CPU work, capacity and duration. The collector does not scan files or tear down workers. |
| `collector.py:JobResourceCollector` | Job snapshot, optional worker totals and traffic to terminal handoff | Owns the participant's task reducer, computes capacity-time, scans retained content once, and observes the workspace filesystem. |
| `cpu_consumed.py`, `message_traffic.py`, `probes/` | OS and existing endpoint observations to measurement objects | Isolate platform and native APIs and keep missing observations local to the affected measurement. |
| `task_rollup.py:TaskResourceReducer` | Registered runtime identities and settled worker files to worker-only subtotals | Deduplicates physical executions, preserves gaps and enforces state and artifact budgets. Launch, cancellation and settlement remain with the runtime. |
| `handoff.py` | Job measurements to/from a checked local terminal file | Publishes atomically, bounds reads and validates each measurement. The parent adds identity separately. |
| `coordinator.py:ResourceStatsCoordinator` | Trusted expected roster and authenticated reports to frozen archive bundle | Accepts one record per participant, closes the roster, publishes records atomically, and identifies missing or invalid participants. |
| `archive_reader.py` | Authorized workspace archive to validated summary/detail | Reads fixed members under separate metadata and decoded-data limits without arbitrary extraction. |

The existing runtime retains lifecycle and authorization ownership. Collection is
placed in entry-point cleanup rather than an application event handler, so an app
construction failure can still produce statistics.

| Runtime integration point | Statistics behavior |
| --- | --- |
| `private/fed/app/client/worker_process.py:main` and `app/server/runner_process.py:main` | Start collection after workspace and custom-path preparation, before application configuration; finalize during owned cleanup. Observe the Client/Server Job, not its persistent parent. |
| `private/fed/app/job_process_cleanup.py:shutdown_job_process_runtime` | Check callback coverage without an extra wait, while preserving the order of publication, streaming shutdown, Cell stop and security close. |
| `private/fed/client/client_app_runner.py:ClientAppRunner.start_run` | After resolving and validating `execution_lifetime`, select task accounting before constructing adapters/components. |
| `private/fed/client/task_worker_executor.py:TaskWorkerExecutor` | Give each supervisor the same participant-owned reducer; preserve saved-result replay and receipt handling. |
| `private/fed/task_worker/worker.py:run_worker` and `supervisor.py:TaskSupervisor` | Collect inside the worker; register before launch and fold after existing settlement proof. |
| `private/fed/client/client_executor.py:JobExecutor._wait_child_process_finish` | Consume the terminal handoff, attach parent-owned identity, release resources through normal owner checks, then report the outcome. |
| `private/fed/server/server_engine.py:ServerEngine.wait_for_complete` | Accept the local Server Job's handoff before removing its terminal process record. |
| `private/fed/server/fed_server.py:process_job_outcome` | Process optional statistics independently of the authenticated job outcome. |
| `private/fed/server/job_runner.py:JobRunner` | Register the deployed roster, finalize before saving the workspace, and retain archive/finalization latches across cleanup retries. |
| `private/fed/server/job_cmds.py:JobCommandModule`, `fuel/flare_api/`, `tool/resources/` | Provide authorized reads of finalized reports, Session methods, and text and JSON rendering. |

Paths in the integration table are relative to `nvflare/`. The statistics package
observes job entry points and lifecycle owners; they retain control of execution,
settlement and authorization.

## Resident job sequence

```text
launcher applies normal environment and resource policy
  -> job entry point prepares workspace and custom import path
  -> collector probes capacity; opens CPU baseline and monotonic window
  -> normal serialization/security setup; application construction and execution
  -> normal shutdown closes command admission; checks unfinished callbacks
  -> before-streaming-shutdown publication:
       freeze CPU/duration and current job-Cell traffic
       scan retained files; observe filesystem; write terminal handoff
       perform existing statistics/workspace upload
  -> existing streaming, Cell, callback and security cleanup
  -> parent observes terminal handle:
       read handoff -> assemble participant -> release resources -> send outcome
```

Capacity probing precedes the measurement window. File scanning and archive
upload follow it. Job-Cell counters are not reset when the collector obtains the
Cell; already observed startup traffic remains included. The final snapshot does
not claim archive-upload or transport-teardown bytes that occur afterward.

Before publication, the runtime checks command callbacks with a zero-time wait.
If callbacks are still active, it marks CPU attribution and traffic observation
incomplete. It retains the existing post-Cell-stop callback drain without delaying
shutdown to improve a measurement. On the client exception path, publication
occurs before MPM's short cleanup grace; the normal owner handles the remaining
teardown.
Collection failure cannot replace the application error or skip subsequent cleanup.

The job process writes `resource_stats/staging/terminal_handoff.json`. The parent
validates it, assigns the job/site identity and consumes/removes it. Resource
release is protected by `finally` and occurs before the existing bounded outcome
request, so server validation or a slow reply does not retain compute resources.
The server parent accepts its own Server Job report locally, without a new network
round trip. A missing handoff yields unavailable observations, not zero.

Handoff reads reject symlinks/nonregular files and enforce the participant byte
limit. These bounds do not interrupt a blocked filesystem syscall. There is no
statistics-specific I/O service or recovery protocol; shared-filesystem latency
remains an operational limitation of this design.

### Client Job restart

A replacement CJ starts new process counters, a new measurement window and a new
worker ledger. The parent retires the earlier terminal handoff before launching
the replacement, so a missing new handoff cannot be mistaken for an old report.
A small attempt-history marker records that prior observations may be missing;
it is not a measurement journal and cannot authorize restart or task replay.

The final report preserves the available terminal generation's subtotal and marks
affected capacity/time, CPU and traffic incomplete when earlier generations were
lost. It does not reconstruct or accumulate all earlier generations. Retained
content is still the final workspace scan. Any duration describes the observed
generation containing the included work, not total wall time across restarts.
Runtime recovery eligibility remains governed by the existing job/task lifecycle.

## Task worker sequence

One `TaskResourceReducer` belongs to a physical Client Job generation and is shared
by all its task adapters. Configuring task mode switches the coordinator to SELF
CPU over its existing baseline; it does not restart the participant clock. Workers
supply their own SELF plus normally waited-descendant CPU. Coordinator CHILDREN
CPU is excluded to avoid counting those workers twice.

```text
supervisor                    worker                         participant collector
register identity/budget
stage private bootstrap
launch; bind returned handle -> claim + authorize bootstrap
                               probe; open shared window
                               build/run/finalize application
                               stage result and checkpoint
                               freeze CPU/capacity snapshot
                               commit result using normal guard
                               close Cell; snapshot traffic
                               publish optional resource file
confirm physical settlement
read/validate/fold once ------------------------------------> worker subtotals
retire resource file
normal result/receipt flow                                    final participant rollup
```

Worker CPU/capacity stop before the completion commit; completion diagnostics
reuse that snapshot. Traffic is taken after normal worker-Cell close. These are
different observation boundaries for different quantities, not separate public
durations. Ordinary Python failures still reach final collection; SIGKILL/OOM
can leave no file. A missing file cannot invalidate a committed task result.

### Private worker record and ledger

`FileTaskArtifactStore` publishes a separate `resource_stats.json` within the
owned attempt directory. Exclusive temporary-file/link publication prevents
replacement; descriptor-relative checks bind access to the expected directory.
The private schema is version `"1.0"` and contains:

- `identity`: the runtime's `job_id`, `site_name`, `task_id`, `task_name`, `attempt_id`;
- `scope: "worker"`, `resource_capacity`, `cpu_consumed`, `message_traffic`;
- optional `duration_seconds`, shared by CPU and capacity for that execution.

The supervisor enables the private `resource_reporting` bootstrap flag only after
reserving a sidecar budget. It binds `TaskHandleSpec.execution_id` in the owner's
ledger, including a handle returned by `TaskLaunchError`. The sidecar does not
supply new execution authority or add fields to result/checkpoint digests.

For singleton workers, the deduplication key is the trusted job/site/task/
attempt tuple plus worker scope; the full identity is checked on every read.
The bound handle supplies lifecycle evidence, not an untrusted file claim.

| Private disposition | Meaning and effect |
| --- | --- |
| `unresolved` | The execution is registered without settlement or proof that it did not launch. The report retains a coverage gap and omits the containing participant duration. Later owner-confirmed settlement may resolve the gap. |
| `never_started` | Launch was never entered, so there is no worker contribution or missing-work claim. An ambiguous launch failure does not qualify for this state. |
| `reported` | A settled worker has a valid record. Each independent measurement is folded in once, even if the training result failed validation. |
| `invalid` / `settled_missing` | Settlement is confirmed but a usable record is absent. Other subtotals are preserved and affected rollups are marked incomplete. |

Equivalent re-reads add nothing. For conflicting records, the reducer retains the
first contribution and marks a gap. It retires the optional file after processing,
retaining its reservation if deletion fails and releasing it once cleanup succeeds.
Identity tombstones must not be evicted in a way that lets the same execution be
admitted as new work.

The per-generation bounds are 64 KiB per sidecar, 8 MiB reserved pending
sidecars, 10,000 scopes, 128 aggregate hardware groups per dimension and 4 MiB
serialized ledger state. The latter bounds encoded state, not Python heap usage.
Limit exhaustion preserves existing totals and marks a gap; it never blocks task
execution. Prior-generation retained files follow ordinary workspace retention
and are outside the current generation's budget.

At participant finalization, the job collector combines the reducer's worker
capacity-time, CPU work and traffic with the coordinator's values. Overflow in one dimension preserves other
valid subtotals. The job collector observes retained content and filesystem capacity once.
Registration after collection starts and settlement before finalization establish
containment without comparing clocks across hosts. Unresolved scopes suppress
the public duration and CPU average while leaving usable totals available.

## Finalization and query integration

The client extends the existing `REPORT_JOB_OUTCOME` request with optional report
bytes. The authenticated server handler returns a separate statistics status;
valid job outcome processing continues if statistics are missing or rejected.
Acceptance runs before the duplicate-outcome gate, so retrying an outcome does
not apply it twice but can retrieve the statistics result.

`JobRunner` registers the server and deployed clients, waits only its existing
outcome grace, then freezes the roster and accepted bytes. The coordinator writes
participant files followed by the summary commit marker before the normal workspace
archive is saved. Per-job locks serialize closure/publication; unrelated jobs do
not wait behind a global filesystem lock. The existing archive-written latch
prevents cleanup retries from rebuilding the archive from partially deleted
sources. Publication failure discards the incomplete optional bundle and follows
normal archive policy.

`Session.get_job_resources(job_id, site=None)` and `get_study_resources()` use
existing admin commands and per-job/study authorization. The archive reader
validates stored identity, reads only fixed statistics members, and derives totals
from recorded `resource_time`. It never falls back to live-process estimates.
Study authorization precedes archive access for each candidate; denied jobs appear
in neither rows nor coverage counts. See the [reporting design](resource_statistics_reporting.md)
for the complete record, retry, error and authorization rules.

`nvflare resources` reuses CLI session, kit and study handling. `--schema` works
offline and describes the command interface; `--format json` uses the standard
success/error envelope. Rendering changes display precision only. The resource
command's default wait is 300 seconds unless the session overrides it; client
timeout does not cancel server storage I/O.

## Backend boundaries and design choices

| Execution profile | Required observation boundary |
| --- | --- |
| Resident Process/container/scheduler job | Inside the actual Client/Server Job after its limits and visibility are applied. Managed trainers retain their existing CPU/traffic attribution. |
| Admitted singleton task backend | The common worker inside its process, Pod, container or allocation; optional artifact access and physical settlement stay with the trusted launcher. |
| Reused/grouped worker | One runtime-issued session identity and one final contribution at session close, including between-assignment idle time. Per-assignment cumulative addition is invalid. |
| External/attach trainer | A runtime-owned or authenticated cooperating scope with an attributable window. Do not count an unrelated preexisting process lifetime. |
| Distributed/multi-node execution | Disjoint CPU scopes and one capacity owner per shared resource boundary. Multi-node accounting requires node-local observations; rank-zero capacity alone cannot represent the other nodes. |

Each profile uses the runtime's session and scope integration and follows its
admission checks. A `TaskLauncherSpec` alone does not establish an attributable
measurement scope. The accounting contract applies to each admitted mode with
such a scope.

The main tradeoffs are deliberate: endpoint counters and final snapshots avoid a
sampling service; explicit capacity-time avoids a public worker inventory; immutable
reports and normal archives avoid a new database; gaps avoid inventing measurements
after crashes. Collection overhead depends on the workload, especially fresh
GPU-probe startup for short tasks. The observation and state limits bound
collection work without changing the runtime's scheduling or cleanup policy.
