# Task groups: a worker span declared by the controller

Status: proposal for 2.10. Builds on #5405 (`execution_lifetime="task"`,
`TaskWorkerExecutor`, `TaskSupervisor`, the task worker and its artifact
store). Companion to `collab_single_workflow_interface_design.md`, where
`with collab.task()` is the same concept on the Collab side. The cellnet
worker transport that an earlier draft of this document bundled in is now
its own proposal, `task_worker_cellnet_session.md`; groups do not depend on
it.

## 1. Problem

#5405 made the client worker's lifetime equal to one task assignment: the
Client Job launches a fresh process per assignment, the process runs
`initialize`, `execute`, `finalize`, and exits. Job lifetime is the other
extreme: one process for the whole job. Nothing in between can be expressed,
because three separate things are fused into "a task":

| Concept | Unit | Today |
| --- | --- | --- |
| Assignment | one task: identity, attempt, receipt, result, checkpoint promotion | the unit of accounting, issued by the server, owned by the Client Job |
| Worker span | the process: launch, `initialize`, N × `execute`, `finalize`, exit | fixed to one assignment (`task`) or the whole job (`job`) |
| Application state | in memory within a span, checkpoint across spans | follows the span |

Workflows that issue several assignments per client and want them served by
one process cannot say so:

- Cross-site evaluation issues one `validate` task per model to every
  client. Under task lifetime each is a worker launch and a model load.
- Federated statistics issues `fed_stats_pre_run` and two `fed_stats`
  passes. #5405 had to require that the generator initialise from local data
  on every assignment.
- Any train-then-validate workflow pays two launches per round.
- SCAFFOLD and local-model cross-site submission were excluded from task
  lifetime because their client state spans assignments.

A task group lets the controller declare the span. Accounting stays per
assignment. The site keeps the last word on resources.

## 2. Programming model

### 2.1 Controller

```python
class CrossSiteModelEval(Controller):
    def control_flow(self, abort_signal, fl_ctx):
        ...
        with self.task_group(fl_ctx):
            for model_name in models:
                self._send_validation_task(model_name, fl_ctx)
            self._wait_for_all_results(abort_signal, fl_ctx)
        # exit: every client that received a task in the block is told the group ended
```

- `Controller.task_group(fl_ctx) -> TaskGroup` is a context manager. Every
  `Task` issued through `broadcast`, `send` or `relay` (and their `_and_wait`
  forms) from the thread that opened the block carries the group's ID.
- `Task(..., group=some_group)` sets it explicitly. Explicit wins over the
  ambient block. Tasks issued from `result_received_cb` and the other task
  callbacks run on server request threads, not the control-flow thread, so
  they are a group of one unless they pass `group=` explicitly.
- A group closes on block exit, including by exception. Issuing a task into
  a closed group raises.
- Groups do not nest. Opening a group inside an open group is an error.
- A task issued outside any block is a group of one, which is exactly
  today's behaviour.

### 2.2 Executor and script author

Inside a group the executor contract is the job-lifetime contract restricted
to the span:

| | one assignment per worker (today) | within a task group |
| --- | --- | --- |
| `initialize(fl_ctx)` | per assignment | once per worker, before the first `execute` |
| `execute(...)` | once | once per assignment of the group |
| `finalize(fl_ctx)` | per assignment | once, at worker end |
| state in `self` | lost after the assignment | kept across the group's assignments served by this worker |
| `flare.system_info()["task_checkpoint"]` | per assignment | per assignment, refreshed before each `execute` |

A Client API script needs no change: it already loops `receive`, `send`
until `flare.is_running()` is false, and `is_running()` turns false at
worker end instead of after one exchange. The #5405 `hello-pt` script runs
as is. Inside a group the script runs on a worker thread, because the
worker's main thread feeds it one assignment at a time; a group of one keeps
#5405's main-thread execution.

A group may contain assignments with different task names (train, then
validate). The checkpoint slot stays per executor and task name, so each
assignment sees its own task's checkpoint.

### 2.3 Site

`local/resources.json` `task_execution` gains one setting:

```json
"task_execution": {
  "worker_timeout": 600,            // unchanged: bounds one attempt's execution
  "group_idle_timeout": 120,        // new: worker ends after this long without a next assignment
  "settlement_timeout": 10, "poll_interval": 0.05, "finalization_timeout": 30,
  "artifact_cleanup": "receipt", "environment_variables": []
}
```

The site cannot extend a group beyond what the controller declared; it can
only shorten it. A worker ended by the site, by abort or by failure looks to
the workflow like any other worker end: the next assignment of the same
group starts a fresh worker. A group therefore spans one or more sequential
workers on a client; it never spans two at once.

## 3. Server side

### 3.1 Identity

`ReservedHeaderKey.TASK_GROUP_ID = "__task_group_id__"`, set as both header
and cookie on task data in `WFCommServer._do_process_task_request`, next to
`TASK_ID` and `TASK_ATTEMPT_ID`. `TaskGroup` (`nvflare/apis/controller_spec.py`)
holds `id` (uuid4) and `closed`. `Task` gains `group: Optional[TaskGroup]`.
The ambient block is a thread-local on the controller, read by
`Controller.broadcast/send/relay` when `task.group is None`.

### 3.2 Open-group table

`WFCommServer` keeps `group id -> clients touched` for the current workflow,
filled when a `ClientTask` of a grouped task is sent. `close_task_group(id)`
moves the touched clients to a per-client "closed, to announce" list and
drops the group. Nothing else is kept per group.

### 3.3 Telling the client the group ended

The Task plane is pull based, so closure is delivered on the client's next
`getTask`. Every reply to `getTask`, whether it carries a task or
`SpecialTaskName.TRY_AGAIN`, gets a header
`ReservedHeaderKey.TASK_GROUPS_CLOSED = "__task_groups_closed__"` listing
closed group IDs that touched this client. Each ID is listed in two
consecutive replies, so one lost reply does not lose the closure; the site
idle timeout covers a client that never polls again. The client treats the
list as idempotent.

Why a header and not a new special task name: `TRY_AGAIN` replies already
carry headers (`TaskConstant.WAIT_TIME`), older clients ignore unknown
headers, and a client that immediately receives a task from a new group
learns in the same reply that the old one ended.

### 3.4 Interaction with the receipt contract

Nothing in `task_assignment_admission.md` changes:

- Resend of an assignment keeps task ID and attempt ID. `TaskWorkerExecutor`
  re-reads the saved result for a completed attempt as today; it never
  relaunches for a resend.
- Receipts (`RECEIVED`, `TASK_CLOSED`, `RETRY`) are per assignment. Payload
  release per assignment on terminal receipt is unchanged; the worker's
  process lifetime is not tied to any of it.
- `process_task_check` and `check_submission` do not look at the group.

### 3.5 Dead clients and abort

A client that goes dead holds no server-side group state worth cleaning:
the touched-client set is bookkeeping for notification only. Job abort ends
workers through the existing `END_RUN` handling in `TaskWorkerExecutor`.

## 4. Client side

### 4.1 `ClientRunner`

`fetch_and_run_one_task` reads `TASK_GROUPS_CLOSED` from every reply and
fires `EventType.TASK_GROUP_CLOSED` with `FLContextKey.TASK_GROUP_ID` once
per listed group. `_do_task` sets `FLContextKey.TASK_GROUP_ID` from the task
header before filters run. Job-lifetime executors may ignore the event;
`TaskWorkerExecutor` acts on it.

`ClientRunner` executes one assignment at a time, so a client's assignments
are contiguous per group as the server issued them.

### 4.2 `TaskWorkerExecutor`

State: at most one open worker session, `(group_id, session)`. On `execute`:

1. If the assignment was completed before, return the saved result, as today.
2. Read `group_id` from the context. No group means a group of one, which
   takes #5405's `TaskSupervisor.run` path unchanged.
3. If a session is open for a different group, close it: the server
   sequences a client's assignments, so a different group means the previous
   one is over even if the closure header has not arrived yet.
4. If no session is open, open one and launch the worker with this attempt
   as its first.
5. Run the attempt: stage input and checkpoint snapshot, hand the attempt to
   the live worker, wait for its completion record, validate identity, PID,
   digest, replay analytics, promote the checkpoint, record completion.
6. Arm the idle timer. It closes the session if no assignment arrives before
   `group_idle_timeout`.

`handle_event`:

- `TASK_GROUP_CLOSED` for the open session's group: close the session.
- `ABORT_TASK`: cancel the active worker; a cancelled worker is killed, not
  reused, and the session is dropped.
- `ABOUT_TO_END_RUN` / `END_RUN`: close any open session, then the existing
  shutdown path.

Closing a session writes the end marker, waits for the worker to finalize
and exit, confirms settlement, and replays the worker log.

### 4.3 `TaskSupervisor`

#5405's `run` (one process per attempt) stays as is. Three methods are
added beside it:

```python
open_session(group_id, store, runtime_root, launcher, request_factory) -> Session
run_attempt(session, bootstrap, input_payload, abort_signal, options, deadline_provider, stage_hook) -> TaskSupervisorOutcome
close_session(session, timeout, options) -> TaskExecutionStatus | None
```

`run_attempt` stages the attempt directory, input and checkpoint snapshot,
then either launches the worker (first attempt: the bootstrap carries the
session directory) or publishes the attempt to the running worker through
the session directory. It waits for the attempt's `completion.json` or
`failure.json`, or for the process to end. A worker that ends mid-attempt is
settled and treated as #5405 treats a worker exit. Per-attempt bookkeeping
(`record_receipt`, `read_saved_result`, payload release, launch identity
replay prevention) is unchanged and keyed by attempt identity.

### 4.4 Worker loop

`run_worker` today: claim, authorise, read input, `initialize`, `execute`,
`finalize`, stage, commit, exit. Session mode:

```
preflight (credential pop, process group, parent guardian, audit, policy check, FOBS) as today
build executor; initialize(fl_ctx) once
loop:
    claim attempt; read input; execute(...); stage result; seal checkpoint; seal analytics; commit completion
    on failure: write failure.json for this attempt, exit the loop
    wait for the next attempt file or the end marker in the session directory
finalize(fl_ctx); exit
```

Each attempt keeps its own directory, `completion.json`, `failure.json`,
result digest and retention, so per-attempt validation in the Client Job is
unchanged. What is new is only that the process outlives one attempt, and
that the checkpoint is sealed after each `execute` rather than after
`finalize`.

### 4.5 Checkpoint

Unchanged from #5405 in mechanism and granularity: one slot per executor and
task name, candidate sealed per attempt, promoted after result validation,
idempotent, released on terminal receipt. Inside a group the application
usually does not need it, because state is in memory; it still needs it
across groups. A group makes two of #5405's exclusions removable:

- SCAFFOLD: control variates live in memory within the group and are
  checkpointed at the end of each `train` attempt for the next group.
- Local-model submission for cross-site evaluation: the trainer saves the
  local model into its `train` checkpoint; the `submit_model` assignment, in
  a later group, reads it.

Those two changes are in the executors, not in the group mechanism, and are
not part of this proposal.

## 5. Session directory: the disk protocol between Client Job and worker

A group needs three things the one-shot worker did not: deliver attempt
N+1's input to a live worker, learn that attempt N completed, and tell the
worker the session is over. Abort (process-group kill) and liveness (process
poll) exist. All three use files, like everything else between the Client
Job and the worker today:

| Direction | File | Writer |
| --- | --- | --- |
| next attempt | `<runtime_root>/sessions/<session>/<seq>.json`, the attempt identity; its input is in the attempt directory as today | Client Job |
| attempt done | `<attempt>/completion.json` or `<attempt>/failure.json`, as today | worker |
| session over | `<runtime_root>/sessions/<session>/end` | Client Job |

The worker polls the session directory at the site poll interval. The
Client Job already polls the process at that interval. Nothing about
payload storage, digests, settlement or retention changes. The worker's log
file is per session, replayed when the session closes.

`task_worker_cellnet_session.md` proposes replacing the per-attempt file
handoff with a cellnet session for large payloads, live analytics and
launchers without a shared filesystem. That proposal keeps this section's
semantics and changes only the transport.

## 6. Packaged workflows that adopt groups

| Controller | Group | Effect under task lifetime |
| --- | --- | --- |
| `CrossSiteModelEval` | around the whole validation phase | one worker per client, one model load per client instead of per model |
| `StatisticsController` | around `pre_run_task_flow` and both `statistics_task_flow` passes | the generator initialises once; the per-assignment re-initialisation requirement from #5405 goes away |
| `FedAvg`, `ScatterAndGather`, `CyclicController` | none | one assignment per round already |
| user controllers | at the author's discretion | `with self.task_group(fl_ctx):` around any train-then-validate round |

`SwarmClientController` and the other client-side controllers are job
lifetime and unaffected.

## 7. Compatibility

- Older clients ignore the group header and the closure header: one worker
  per assignment, correct and slower. Older servers send no header: same.
  No `format_version` change is required; #5405's version 3 gate for task
  lifetime stays as it is.
- Job-lifetime clients see the headers and the `TASK_GROUP_CLOSED` event and
  may ignore both.
- `execution_lifetime="job"` remains the default. Groups change nothing for
  jobs that do not opt into task lifetime.

## 8. Relationship to Collab

`with collab.task()` in the Collab design is this mechanism with the call as
the assignment: implicit membership of every call inside the block, one
worker per site per block, closure on block exit, site bounds. One
correction follows from this proposal: Collab should use the same lease
model, per-call attempts under one worker, rather than treating the block
as a single attempt. One accounting model on both planes, and the Collab
block then needs nothing the Task plane group does not already have.

## 9. Tests

- Unit, `Controller` and `WFCommServer`: group stamped on tasks inside the
  block and not outside; explicit `Task(group=)` wins; closure on exit and
  exception; closed-group list appears in task and `TRY_AGAIN` replies for
  two requests; issuing into a closed group raises; nesting raises.
- Unit, `TaskWorkerExecutor` with the real worker: group of one behaves
  exactly as #5405; two assignments of one group reach one worker with
  `initialize` once and `finalize` once; a different group closes the open
  session; `TASK_GROUP_CLOSED` closes it; idle timeout closes it; resend of
  a completed attempt re-reads the saved result; checkpoint promoted per
  attempt.
- Worker loop: two attempts in one process with per-attempt completion
  records; failure in attempt two leaves attempt one's completion intact and
  ends the session.

## 10. Decisions

1. A task from a different group implicitly closes the open session. The
   server sequences a client's assignments, and two workers on one client
   is a resource surprise.
2. `group_idle_timeout` defaults to 120 s: long enough for server-side
   aggregation between a round's `train` and `validate`, short enough that an
   abandoned group releases resources.
3. Both the ambient block and explicit `Task(group=)` ship; callbacks on
   other threads need the explicit form.
4. Checkpoint slot stays per executor and task name.
