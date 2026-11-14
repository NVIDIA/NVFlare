# Automatic client-job recovery

CP automatically replaces a failed CJ using the same job ID, launcher, metadata,
resource reservation, and persistent job workspace. The replacement uses normal
`GET_TASK` delivery: the unchanged server resends an outstanding assignment with
its existing task and attempt IDs. User code restores its own checkpoint.

Implemented for eligible resident (`execution_lifetime="job"`) Process, Docker
and Slurm client jobs, and for Kubernetes client jobs in job or task lifetime.
Kubernetes requires the persistent jobs PVC profile and confirmed shutdown of the
CJ and its workers. See the
[Kubernetes persistence and resume contract](../user_guide/admin_guide/deployment/kubernetes_task_launcher.rst#persistent-files-and-cj-resume).
Local Process/Docker task execution is deferred; the
[retired Process task-mode eligibility gap](#process-task-mode-recovery-gap) is described below.
CP and the server must remain running.
Hierarchical federation recovery is outside this feature's scope. Intermediate
CJs retain their existing failure behavior because replacement cannot restore
their in-memory child-result receipts; automatic recovery is limited to leaf CJs.
The job's client registration must also remain valid: re-registering CP creates a
new token that the existing server job does not adopt. Migrating that registration
is outside this client-only recovery feature.

## Recovery setting

`max_cj_restart_failures` is configured on CP, with default `3`.
Set `NVFLARE_MAX_CJ_RESTART_FAILURES` in CP's environment before startup, or add
`max_cj_restart_failures=<value>` to the `--set` arguments of CP's Python launch
command. `--set` takes precedence over the environment variable.
`0` disables recovery. No application opt-in is required.

The limit counts consecutive attempts to get a replacement initialized. There is
no lifetime restart limit. CP increments the counter when it launches a replacement
and resets it when that CJ completes runner initialization and reports READY.
The original failure does not consume an attempt. Three replacement startups that
fail consecutively end recovery; a successful replacement restores the full budget.
A two-day task can therefore span twelve or more four-hour allocations without
exhausting the budget, provided its checkpoint and server assignment remain valid.

This measures initialization success, not training progress. A CJ that repeatedly
initializes successfully and then crashes can continue restarting. Application
checkpoint compatibility and progress remain the application's responsibility.

## Allocation wall-time and task lifetime

In Kubernetes task mode, a confirmed worker crash or scheduler timeout exits CJ without
closing the server assignment. CP waits for CJ and its workers to stop, then
uses the same restart policy. A valid committed result takes precedence;
application errors, rejected submissions and cancellation do not trigger recovery.

Kubernetes supports this path with the persistent jobs PVC. Slurm
([task adapter](https://github.com/NVIDIA/NVFlare/pull/5513)) uses the shared
workspace: a worker allocation settling as TIMEOUT, PREEMPTED, OUT_OF_MEMORY,
NODE_FAIL, BOOT_FAIL or a signal-killed FAILED exits CJ, and CP replaces it after all workers settle. No
signal precedes the wall-time limit, so work since the last checkpoint repeats. Local Process/Docker
task mode is deferred because its CJ restart gates do not establish worker
settlement; see the [eligibility gap](#process-task-mode-recovery-gap) below.

### Wall-limit and owner-loss timeline

| Event | Recovery behavior |
| --- | --- |
| Kubernetes worker allocation expires | After confirmed worker settlement, CJ exits and CP replaces it using the existing recovery policy. Application code restores its partial checkpoint. |
| Kubernetes CJ dies while a worker may still exist | CP retains cleanup ownership and replaces CJ only after both executions have stopped. |
| Result reply is lost while CJ remains alive | CJ retransmits its saved result without launching another worker. |
| CP dies | Ownership adoption is unsupported; persistent files alone do not authorize replacement. |

Replacement CJs derive fresh physical worker IDs while normal server delivery
preserves an outstanding assignment's IDs. Existing assignment deadlines still
apply; recovery does not extend them.

### Native task-resume follow-up requirements

Partial-progress checkpoints remain application-owned. Applications must publish
complete, compatible state that survives abrupt termination; the framework's
completion-bound checkpoint handoff does not periodically save unfinished work.
Slurm task-adapter integration, GPU recovery and Lustre qualification remain
separate work. CP restart adoption remains outside this feature.

## Implementation

- `JobExecutor` keeps its existing logical job entry through replacement and
  queueing, so CP's ordinary heartbeats continue listing the job. It uses one
  monitor and reports a terminal outcome/releases resources once, at the end.
- Physical launch reuses the original launch specification with current CP
  bootstrap tokens. The per-job certificate stays until the logical job ends;
  application files and checkpoints are preserved without redeployment.
- STARTED still marks command availability before initialization, preserving
  graceful aborts during startup. READY resets the restart counter after
  `ClientRunner.init_run`; CP continues exposing STARTED as the job status.
  Notifications retry within the existing timeout without failing the CJ.
  A private CP/CJ launch ID rejects mismatched notifications from old executions
  when provided. Notifications without an ID remain accepted for compatibility
  with older CJs and custom launchers. Both launcher option representations carry
  it, including Docker and Kubernetes.
  STOPPED cannot regress. There is no new server protocol.
- Job abort cancels pending/in-flight replacement through the existing pending
  handle. A task abort during replacement or after confirmed death aborts the job.
  Otherwise it retains the existing best-effort command behavior and does not
  permanently disable recovery. Task-abort delivery is not journaled or replayed.

## Confirmed failures only

`JobHandleSpec.can_restart()` supplies termination evidence; unsupported launchers
return false. CP separately checks the failure code, cancellation, completion,
and consecutive-attempt limit.

| Backend | Required evidence |
| --- | --- |
| Process | The launcher-owned CJ has exited, its process group is absent, and no external-trainer launch bootstrap remains. |
| Docker | Container wait/inspection confirms exit, rather than merely losing its record. |
| Slurm | Accounting reports COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY, or PREEMPTED, and the allocation has left the active queue. |
| Kubernetes | The same CJ Pod UID has confirmed container/sandbox shutdown, its PVC workspace finished initialization, and all CP-owned task workers have settled. |

Normal success, STOPPED teardown, cancellation, configuration errors, and unsafe
component failures do not trigger recovery. Missing accounting, node loss, backend
API errors, and network/heartbeat failures are not proof of termination. A launch
exception ends recovery rather than risking duplicate submission when acceptance
is unknown. If a handle was published or returned before launch setup failed, CP
attempts cancellation and observation before releasing its resources. Unconfirmed
cleanup retains ownership as described below. Docker physical launches use
unique container names, allowing delayed auto-removal of the previous container.
Existing scheduler queue/startup timeouts remain in effect.

Process recovery is skipped when an `external_process` trainer launch bootstrap
remains in the app directory. The backend writes this existing file before
launching a detached trainer and removes it only after confirming trainer-group
exit. After an abrupt CJ loss, the file remains even if the trainer subsequently
exits, so CP ends the job instead of risking overlapping trainers. The existing
trainer watchdog handles detached-group termination. Docker and Slurm retain
recovery when the container or allocation is confirmed dead.

After a failed CJ exits, the Process handle waits up to 10 seconds for
its owned group to drain, kills remaining members of that group, and waits up to
10 more seconds to confirm exit. Recovery requires the group to be gone; cleanup
timeout or uncertain observations end recovery. Job/task cancellation during
cleanup prevents replacement. No restart-blocker file or new orphan-process
manager is introduced.
Slurm/container recovery assumes training remains inside the managed allocation
or container. CP is the restart owner; independent scheduler/container requeue
policies must not compete with it.

Kubernetes task mode uses the jobs PVC; resident CJs require
`persist_resident_workspaces`. The PVC replaces the workspace `emptyDir`.
The first CJ downloads into it; replacements reuse it to preserve checkpoints.
If shutdown exceeds `cleanup_timeout` (60 seconds for resident CJs), recovery
stops while CP continues worker cleanup. Replacement Pods get fresh launch IDs,
task capabilities and credential Secrets; the old task capability is closed.
Old CJ Pod deletion is best effort; the final Pod is retained for diagnostics.

Native worker attempt IDs include the CJ launch ID; server task and attempt IDs
stay unchanged. Logs map server attempts to physical worker IDs.

### Process task-mode recovery gap

In the retired Process task implementation, each task worker started a separate
process group. `ProcessHandle.can_restart()` checks the CJ's own group and
`client_api_bootstrap_*.json` external-trainer files; it does not check task-worker
groups or exclude task lifetime. Consequently, CP could find CJ replacement eligible
without proof that its task workers had stopped. The worker's startup parent-PID
check was not CP-owned settlement evidence after a later CJ failure.

This eligibility gap motivated deferring local Process/Docker task execution.
Shared client-parent admission now rejects local task-lifetime jobs before
launching a CJ. The deferral adds no task-worker recovery owner and leaves
resident restart gates unchanged. CJ-group exit alone must not be used as proof
that an independently launched worker has settled.
See the [backend/lifetime matrix](task_worker_lifecycle.md#backend-and-recovery-boundaries).

## Failed launch cleanup and retained ownership

This client-parent behavior applies to initial and replacement CJ launches whose
launcher publishes an accepted handle through `publish_job_handle()` or returns
one. CP attaches that handle before applying a saved abort. A launch exception,
a missing return handle after publication, or an AFTER_JOB_LAUNCH failure triggers
one cancellation attempt followed by one completion observation. Cancellation
failure cannot skip observation. Failure to start the completion monitor uses the
same cleanup path.

`job_launch_cleanup_timeout` bounds each launch-cleanup control wait (default
10 seconds, finite and positive). Configure it on CP through
`NVFLARE_JOB_LAUNCH_CLEANUP_TIMEOUT` or `--set job_launch_cleanup_timeout=<seconds>`;
`--set` takes precedence. A saved abort has its own bounded wait; cleanup reuses
that cancellation rather than starting another. Control calls and the ordinary
CJ monitor use daemon threads so an unavailable backend cannot indefinitely hold
CP process exit. This does not make a runtime disappear when CP exits.
Slurm cancellation waits requery at most one second apart, even with the default
10-second running-job poll interval, so a normally cancelled allocation can be
observed within the default parent cleanup bound. Query outages or slow scheduler
settlement can still exceed that bound and require reconciliation.

If cancellation is still running, observation raises, or observation exceeds its
bound, CP logs a critical reconciliation message and keeps the run registration,
handle, control-call state, admission allocation/token/resource-manager reference,
and job credentials. The start request returns an error without freeing that
allocation. Replacement failure keeps the same owner. Ordinary monitor wait/query
errors also preserve the registration and credentials. Retained jobs stay in CP's
job list and reject another launch of the same logical job, including runs whose
CJ already reported STARTED. A repeated start releases only its new admission
allocation; the retained run keeps its original allocation. Status queries expose
`cleanup pending`. Deletion through the client parent rejects every registered
run, preserving credentials and launch files even if the server has forgotten it.
The original launch failure is logged before cleanup errors.

An explicit admin abort still sends a cooperative abort and makes a bounded
cancellation/observation attempt. It reuses in-flight cleanup calls, and another
explicit request may retry completed failed calls. Concurrent admin requests do
not stack control operations. Heartbeats neither retry retained cleanup nor emit
repeated warnings. A different start request that fails before acquiring a runtime
still frees its own admission allocation.

There is no automatic settlement retry loop or finalization of retained runs.
An in-flight control call may later finish, but the run remains registered for
operator reconciliation. A normal `JobHandleSpec.wait()` return is the launcher's
existing completion contract; this change does not strengthen each backend's
settlement evidence or claim GPU release. A failed cancellation followed by a
successful completion observation preserves the original launch error and permits
release. No cleanup error fabricates an application failure or authorizes another
submission. Verified task results retain their existing success semantics; task
outcome, cancellation intent, scheduler settlement and GPU availability are
separate facts.

### Operator reconciliation and parent loss

1. Keep the retained run's workspace, launch files and credentials intact. Preserve
   the CP logs and the accepted handle's exact backend identity. Do not resubmit,
   remove the run, or restart CP to clear its admission reservation while its
   runtime may survive. If the control calls have not returned, involve the backend
   operator rather than issuing competing cleanup operations.
2. Have the backend operator establish ownership before any destructive action.
   For Slurm, validate the exact allocation ID, submitting UID, job name, comment
   marker, submission time, and command/work-directory paths against this site's
   run and launch records. A shared job name or `nvfl:<job_id>` marker alone is
   insufficient. After CP loss or a lost submission reply, candidate lookup must
   reject foreign or ambiguous matches; missing records during accounting lag or
   an outage do not establish rejection. Never cancel by a broad name match or
   blindly repeat `sbatch`.
3. Cancel only the verified owned runtime when necessary, then obtain backend
   settlement evidence. `scancel` acknowledgement, elapsed time, a successful task
   result, and a CJ's STOPPED notification do not prove allocation or GPU release.
   Check the exact allocation's active queue/accounting records and, for unresolved
   node/step cleanup, obtain scheduler/node-admin confirmation before declaring
   capacity available. Preserve unresolved identity/files and keep the affected
   capacity out of admission while evidence is unavailable.
4. Once every retained runtime is reconciled, the operator may remove its stale
   per-job credentials/control artifacts and rebuild the affected admission state
   according to the site's resource manager. Restart CP only after this audit,
   including any CP-owned task-worker allocations. There is no supported live
   command to finalize a retained registration (admin abort requests cancellation
   and observation, not finalization), and CP restart adoption is a
   separate feature. If CP is lost, this reconciliation responsibility transfers
   to the operator; in-memory ownership is not durable adoption.

### Resident Slurm submission gap and qualification

The resident `SlurmJobManager` currently registers its handle only after `sbatch`
returns a usable ID. An accepted submission whose reply times out, raises, or lacks
an ID can therefore fail before publication; its current exception path removes
launch files. `ClientSlurmJobLauncher` also releases the task owner when that CJ
launch raises. These paths are **not fixed by caller retention**. The server launch
path does not install JOB_HANDLE_CALLBACK and registers SJ only after launch returns;
this client change claims no SJ ownership fix.

A separate Slurm slice must register/persist an uncertain submission identity
before invoking `sbatch`, retain its files, avoid blind retry, reconcile only exact
owned identities, and give both CP and SP explicit bounded cleanup ownership.
Nonzero/signal exits and missing replies are not automatically rejection. That
slice must review the existing resident terminal/missing-record settlement rules,
which are weaker than the CP-owned task submission path. It must preserve task
success when allocation cleanup remains unconfirmed.

Unit fault injection verifies caller policy, not scheduler or GPU behavior. Live
Slurm qualification must exercise accepted-but-lost replies, cancellation/query
outages, controller/accounting lag, allocation/step teardown and physical GPU
release for CJ and SJ as applicable. Earlier operator-inventory hardware runs do
not qualify this change or the pending Slurm slice.

### Kubernetes settlement limitation

The current `K8sJobHandle.terminate()` marks the handle terminal and releases its
job resources even when Pod deletion raises an API or network error. Its `wait()`
then returns without confirming Pod/container shutdown. The caller can therefore
release ownership despite a surviving Pod; this PR does not fix that backend
contract. Track the separate [Kubernetes ownership follow-up](https://github.com/NVIDIA/NVFlare/issues/5791).
Do not describe the caller-retention fix as Kubernetes settlement protection.

## Task and checkpoint boundaries

Native task checkpoints atomically record the logical assignment and its saved
result. If CJ dies after promotion, its replacement verifies and resends that
result without advancing the checkpoint again; missing result artifacts fail
the assignment instead of rerunning it from already-advanced state.

No task journal, model-input cache, result outbox, or checkpoint API is added.
If the server already accepted the result, its existing receipt handling applies.
If a replacement cannot synchronize with its server job within the existing
timeout, it uses the existing server-parent GET_CLIENTS query to confirm whether
that job has ended. Only an explicit response for this job saying it is no longer
active lets the replacement finish normally; communication errors remain errors.
If the task expired or the workflow moved on, normal task delivery governs what
the new CJ receives. Existing task deadlines must cover training and queue delays.

Task-send callbacks and filters may run again, so this is not byte-identical replay
of arbitrary filters or exactly-once execution. Checkpoints must match the received
task/round. Work since the last complete checkpoint can repeat.

## Validation

The automated provisioned-process test kills CJ four times during one task,
exceeding the default failed-start limit. Five distinct CJs reuse the same
job/task/attempt IDs, restore an on-disk checkpoint, and produce the uninterrupted
arithmetic result with one server result receipt. Unit coverage also exercises
twelve consecutive successful replacements.
An external-process regression kills CJ while its detached trainer ignores
SIGTERM and verifies that CP ends the job without launching another trainer.
The server implementation is unchanged.

Focused tests cover repeated successful replacements, exhausted/reset failure
budgets, stale status messages, cancellation during launch, unchanged terminal
outcomes, credentials/resources, launcher options, confirmed server-job closure,
and backend termination evidence.

Sources: [CP lifecycle](../../nvflare/private/fed/client/client_executor.py),
[CJ readiness](../../nvflare/private/fed/client/client_app_runner.py),
[existing task redelivery](../../nvflare/apis/impl/wf_comm_server.py),
[executable acceptance test](../../tests/integration_test/fast/cj_recovery_e2e_test.py).
