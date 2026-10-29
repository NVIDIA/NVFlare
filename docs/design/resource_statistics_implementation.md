# Resource statistics: runtime integration

This page describes the components and lifecycle of the format-4 design introduced
in the [overview](resource_statistics_design.md). The [measurement design](resource_statistics_measurement.md)
defines the algorithms; the [reporting design](resource_statistics_reporting.md)
defines records, validation, delivery and queries. The internal interfaces below
keep resource accounting separate from runtime control and application code.

## Components and ownership

The internal `nvflare/private/fed/resource_stats/` package owns collection and reporting.
Its interfaces are internal integration points, not extension APIs for job code.

| Component | Input and result | Responsibility |
| --- | --- | --- |
| `contract.py` | Untrusted JSON/measurements to normalized records; participant entries to job/study totals | Version, shape, numeric and size validation; decimal arithmetic and quality propagation. No runtime or filesystem ownership. |
| `accumulator.py:ResourceCapacityRecorder` | One probe snapshot to normalized startup capacity | Preserve independent dimensions and consolidate equivalent hardware groups. No separate timer. |
| `collector.py:ExecutionResourceCollector` | Finalized execution environment to one immutable `stop()` snapshot | Probe once, open the shared CPU/capacity window, then freeze CPU work, capacity and duration. No file scan or worker teardown. |
| `collector.py:JobResourceCollector` | Job snapshot, optional worker totals and traffic to terminal handoff | Own the participant's task reducer; compute capacity-time; scan retained content once and observe the workspace filesystem. |
| `cpu_consumed.py`, `message_traffic.py`, `probes/` | OS and existing endpoint observations to measurement objects | Isolate platform/native APIs and keep missing observations local to the affected measurement. |
| `task_rollup.py:TaskResourceReducer` | Registered runtime identities and settled worker files to worker-only subtotals | Deduplicate physical executions, preserve gaps and enforce state/artifact budgets. No launch, cancellation or settlement decisions. |
| `handoff.py` | Job measurements to/from a checked local terminal file | Atomic publication, bounded reads and per-measurement validation. Parent-owned identity is added separately. |
| `coordinator.py:ResourceStatsCoordinator` | Trusted expected roster and authenticated reports to frozen archive bundle | One accepted record per participant, closure, atomic publication and missing/invalid roster entries. |
| `archive_reader.py` | Authorized workspace archive to validated summary/detail | Read fixed members under independent metadata and decoded-data limits. No arbitrary extraction. |

The existing runtime retains lifecycle and authorization ownership. Collection is
placed in entry-point cleanup rather than an application event handler, so an app
construction failure can still produce statistics.

| Runtime integration point | Statistics behavior |
| --- | --- |
| `private/fed/app/client/worker_process.py:main` and `app/server/runner_process.py:main` | Start collection after workspace and custom-path preparation, before application configuration; finalize during owned cleanup. Observe the Client/Server Job, not its persistent parent. |
| `private/fed/app/job_process_cleanup.py:shutdown_job_process_runtime` | Check callback coverage without an extra wait; preserve publication, streaming shutdown, Cell stop and security-close order. |
| `private/fed/client/client_app_runner.py:ClientAppRunner.start_run` | After resolving and validating `execution_lifetime`, select task accounting before constructing adapters/components. |
| `private/fed/client/task_worker_executor.py:TaskWorkerExecutor` | Give each supervisor the same participant-owned reducer; preserve saved-result replay and receipt handling. |
| `private/fed/task_worker/worker.py:run_worker` and `supervisor.py:TaskSupervisor` | Collect inside the worker; register before launch and fold after existing settlement proof. |
| `private/fed/client/client_executor.py:JobExecutor._wait_child_process_finish` | Consume the terminal handoff, attach parent-owned identity, release resources through normal owner checks, then report the outcome. |
| `private/fed/server/server_engine.py:ServerEngine.wait_for_complete` | Accept the local Server Job's handoff before removing its terminal process record. |
| `private/fed/server/fed_server.py:process_job_outcome` | Process optional statistics independently of the authenticated job outcome. |
| `private/fed/server/job_runner.py:JobRunner` | Register the deployed roster, finalize before saving the workspace, and retain archive/finalization latches across cleanup retries. |
| `private/fed/server/job_cmds.py:JobCommandModule`, `fuel/flare_api/`, `tool/resources/` | Authorized finalized-report reads, Session methods and human/JSON rendering. |

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

Before publication, check command callbacks with a zero-time wait. If callbacks
are still active, mark CPU attribution and traffic observation incomplete. Keep
the existing post-Cell-stop callback drain; do not delay shutdown to improve a
measurement. The client's exception path performs publication before entering
MPM's short cleanup grace, then leaves remaining teardown with its normal owner.
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
| `unresolved` | Registered, without settlement or proof of no launch. Keep a coverage gap; omit the containing participant duration. Later owner-confirmed settlement may resolve it. |
| `never_started` | Launch was never entered. No worker contribution and no missing-work claim. An ambiguous launch failure is not this state. |
| `reported` | A settled worker has a valid record. Fold each independent measurement once, even if its training result failed validation. |
| `invalid` / `settled_missing` | Settlement is confirmed but a usable record is absent. Preserve other subtotals and mark affected rollups incomplete. |

Equivalent re-reads add nothing. Conflicting records retain the first contribution
and mark a gap. Retire the optional file after processing; retain its reservation
if deletion fails, and release it only once when cleanup succeeds. Never evict
identity tombstones and then admit the same execution as new work.

The fixed per-generation bounds are 64 KiB per sidecar, 8 MiB reserved pending
sidecars, 10,000 scopes, 128 aggregate hardware groups per dimension and 4 MiB
serialized ledger state. The latter bounds encoded state, not Python heap usage.
Limit exhaustion preserves existing totals and marks a gap; it never blocks task
execution. Prior-generation retained files follow ordinary workspace retention
and are outside the current generation's budget.

At participant finalization, fold worker capacity-time, CPU work and traffic into
the coordinator's values. Overflow in one dimension preserves other valid
subtotals. Retained content/filesystem are observed once by the job collector.
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
not wait behind a global filesystem lock. Preserve the existing archive-written
latch so cleanup retries cannot rebuild from partially deleted sources. Publication
failure discards the incomplete optional bundle and follows normal archive policy.

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
| Distributed/multi-node execution | Disjoint CPU scopes and one capacity owner per shared resource boundary; node-local observations rather than rank-zero capacity multiplied by ranks/nodes. |

Each profile requires the runtime's session/scope integration and admission
checks. A `TaskLauncherSpec` alone does not establish that support. The accounting
contract applies to every admitted mode with attributable measurement scopes.

The main tradeoffs are deliberate: endpoint counters and final snapshots avoid a
sampling service; explicit capacity-time avoids a public worker inventory; immutable
reports and normal archives avoid a new database; gaps avoid inventing measurements
after crashes. Collection overhead depends on the workload, especially fresh
GPU-probe startup for short tasks. The observation and state limits bound collection work without
changing the runtime's scheduling or cleanup policy.
