# Resource statistics: runtime integration

This page describes the components and lifecycle of the format-4 design introduced
in the [overview](resource_statistics_design.md). The [measurement design](resource_statistics_measurement.md)
defines the algorithms; the [reporting design](resource_statistics_reporting.md)
defines records, validation, delivery and queries. This docs-only proposal adds no
runtime code or commands to main. The interfaces below describe the intended
implementation, informed by the resource-statistics development branch.

## Components and ownership

The design adds an internal `nvflare/private/fed/resource_stats/` package. Its
interfaces are internal integration points, not extension APIs for job code.

<table class="docutils">
<thead>
<tr>
  <th>Component</th>
  <th>Input and result</th>
  <th>Responsibility</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>contract.py</code></td>
  <td>Untrusted JSON/measurements to normalized records; participant entries to job/study totals</td>
  <td>Version, shape, numeric and size validation; decimal arithmetic and quality propagation. No runtime or filesystem ownership.</td>
</tr>
<tr>
  <td><code>accumulator.py:ResourceCapacityRecorder</code></td>
  <td>One probe snapshot to normalized startup capacity</td>
  <td>Preserve independent dimensions and consolidate equivalent hardware groups. No separate timer.</td>
</tr>
<tr>
  <td><code>collector.py:ExecutionResourceCollector</code></td>
  <td>Finalized execution environment to one immutable <code>stop()</code> snapshot</td>
  <td>Probe once, open the shared CPU/capacity window, then freeze CPU work, capacity and duration. No file scan or worker teardown.</td>
</tr>
<tr>
  <td><code>collector.py:JobResourceCollector</code></td>
  <td>Job snapshot, optional worker totals and traffic to terminal handoff</td>
  <td>Own the participant's task reducer; compute capacity-time; scan retained content once and observe the workspace filesystem.</td>
</tr>
<tr>
  <td><code>cpu_consumed.py</code>, <code>message_traffic.py</code>, <code>probes/</code></td>
  <td>OS and existing endpoint observations to measurement objects</td>
  <td>Isolate platform/native APIs and keep missing observations local to the affected measurement.</td>
</tr>
<tr>
  <td><code>task_rollup.py:TaskResourceReducer</code></td>
  <td>Registered runtime identities and settled worker files to worker-only subtotals</td>
  <td>Deduplicate physical executions, preserve gaps and enforce state/artifact budgets. No launch, cancellation or settlement decisions.</td>
</tr>
<tr>
  <td><code>handoff.py</code></td>
  <td>Job measurements to/from a checked local terminal file</td>
  <td>Atomic publication, bounded reads and per-measurement validation. Parent-owned identity is added separately.</td>
</tr>
<tr>
  <td><code>coordinator.py:ResourceStatsCoordinator</code></td>
  <td>Trusted expected roster and authenticated reports to frozen archive bundle</td>
  <td>One accepted record per participant, closure, atomic publication and missing/invalid roster entries.</td>
</tr>
<tr>
  <td><code>archive_reader.py</code></td>
  <td>Authorized workspace archive to validated summary/detail</td>
  <td>Read fixed members under independent metadata and decoded-data limits. No arbitrary extraction.</td>
</tr>
</tbody>
</table>

The existing runtime retains lifecycle and authorization ownership. Collection is
placed in entry-point cleanup rather than an application event handler, so an app
construction failure can still produce statistics.

<table class="docutils">
<thead>
<tr>
  <th>Existing integration point</th>
  <th>Added behavior</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>private/fed/app/client/worker_process.py:main</code> and <code>app/server/runner_process.py:main</code></td>
  <td>Start collection after workspace and custom-path preparation, before application configuration; finalize during owned cleanup. Observe the Client/Server Job, not its persistent parent.</td>
</tr>
<tr>
  <td><code>private/fed/app/job_process_cleanup.py:shutdown_job_process_runtime</code></td>
  <td>Check callback coverage without an extra wait; preserve publication, streaming shutdown, Cell stop and security-close order.</td>
</tr>
<tr>
  <td><code>private/fed/client/client_app_runner.py:ClientAppRunner.start_run</code></td>
  <td>After resolving and validating <code>execution_lifetime</code>, select task accounting before constructing adapters/components.</td>
</tr>
<tr>
  <td><code>private/fed/client/task_worker_executor.py:TaskWorkerExecutor</code></td>
  <td>Give each supervisor the same participant-owned reducer; preserve saved-result replay and receipt handling.</td>
</tr>
<tr>
  <td><code>private/fed/task_worker/worker.py:run_worker</code> and <code>supervisor.py:TaskSupervisor</code></td>
  <td>Collect inside the worker; register before launch and fold after existing settlement proof.</td>
</tr>
<tr>
  <td><code>private/fed/client/client_executor.py:JobExecutor._wait_child_process_finish</code></td>
  <td>Consume the terminal handoff, attach parent-owned identity, release resources through normal owner checks, then report the outcome.</td>
</tr>
<tr>
  <td><code>private/fed/server/server_engine.py:ServerEngine.wait_for_complete</code></td>
  <td>Accept the local Server Job's handoff before removing its terminal process record.</td>
</tr>
<tr>
  <td><code>private/fed/server/fed_server.py:process_job_outcome</code></td>
  <td>Process optional statistics independently of the authenticated job outcome.</td>
</tr>
<tr>
  <td><code>private/fed/server/job_runner.py:JobRunner</code></td>
  <td>Register the deployed roster, finalize before saving the workspace, and retain archive/finalization latches across cleanup retries.</td>
</tr>
<tr>
  <td><code>private/fed/server/job_cmds.py:JobCommandModule</code>, <code>fuel/flare_api/</code>, <code>tool/resources/</code></td>
  <td>Authorized finalized-report reads, Session methods and human/JSON rendering.</td>
</tr>
</tbody>
</table>

Paths in the integration table are relative to `nvflare/`. The statistics package
and resource commands are additions; the job entry points and lifecycle owners
already exist.

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

For current singleton workers, the deduplication key is the trusted job/site/task/
attempt tuple plus worker scope; the full identity is checked on every read.
The bound handle supplies lifecycle evidence, not an untrusted file claim.

<table class="docutils">
<thead>
<tr>
  <th>Private disposition</th>
  <th>Meaning and effect</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>unresolved</code></td>
  <td>Registered, without settlement or proof of no launch. Keep a coverage gap; omit the containing participant duration. Later owner-confirmed settlement may resolve it.</td>
</tr>
<tr>
  <td><code>never_started</code></td>
  <td>Launch was never entered. No worker contribution and no missing-work claim. An ambiguous launch failure is not this state.</td>
</tr>
<tr>
  <td><code>reported</code></td>
  <td>A settled worker has a valid record. Fold each independent measurement once, even if its training result failed validation.</td>
</tr>
<tr>
  <td><code>invalid</code> / <code>settled_missing</code></td>
  <td>Settlement is confirmed but a usable record is absent. Preserve other subtotals and mark affected rollups incomplete.</td>
</tr>
</tbody>
</table>

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

<table class="docutils">
<thead>
<tr>
  <th>Execution profile</th>
  <th>Required observation boundary</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Resident Process/container/scheduler job</td>
  <td>Inside the actual Client/Server Job after its limits and visibility are applied. Managed trainers retain their existing CPU/traffic attribution.</td>
</tr>
<tr>
  <td>Admitted singleton task backend</td>
  <td>The common worker inside its process, Pod, container or allocation; optional artifact access and physical settlement stay with the trusted launcher.</td>
</tr>
<tr>
  <td>Reused/grouped worker</td>
  <td>One runtime-issued session identity and one final contribution at session close, including between-assignment idle time. Per-assignment cumulative addition is invalid.</td>
</tr>
<tr>
  <td>External/attach trainer</td>
  <td>A runtime-owned or authenticated cooperating scope with an attributable window. Do not count an unrelated preexisting process lifetime.</td>
</tr>
<tr>
  <td>Distributed/multi-node execution</td>
  <td>Disjoint CPU scopes and one capacity owner per shared resource boundary; node-local observations rather than rank-zero capacity multiplied by ranks/nodes.</td>
</tr>
</tbody>
</table>

The last three rows require their runtime's session/scope integration. They are
not enabled by having a `TaskLauncherSpec` or by this documentation. The accounting
contract applies to every admitted mode, while runtime support and deployment
qualification remain prerequisites for claiming support.

The main tradeoffs are deliberate: endpoint counters and final snapshots avoid a
sampling service; explicit capacity-time avoids a public worker inventory; immutable
reports and normal archives avoid a new database; gaps avoid inventing measurements
after crashes. The remaining engineering qualification is actual backend/GPU
execution and collection overhead, especially fresh GPU-probe startup on short tasks.
Those checks establish release readiness; they do not change this data model.
