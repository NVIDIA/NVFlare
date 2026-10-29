# Automatic client-job recovery

CP automatically replaces a failed CJ using the same job ID, launcher, metadata,
resource reservation, and persistent job workspace. The replacement uses normal
`GET_TASK` delivery: the unchanged server resends an outstanding assignment with
its existing task and attempt IDs. User code restores its own checkpoint.

Implemented for eligible resident (`execution_lifetime="job"`) Process, Docker
and Slurm client jobs. Do not infer recovery of independently launched task
workers from those CJ restart checks; the
[Process task-mode eligibility gap](#process-task-mode-recovery-gap) is described below.
Kubernetes does not opt into automatic CJ recovery: its job handle retains the
default `can_restart=False`. The persistent jobs-PVC profile now preserves task
workspaces, and optionally resident workspaces, but it does not adopt retained
directories or reconstruct ownership of earlier workers. See the
[Kubernetes persistence and resume contract](../user_guide/admin_guide/deployment/kubernetes_task_launcher.rst#persistent-files-and-cj-resume).
CP and the server must remain running.
Hierarchical federation recovery is outside this feature's scope. Intermediate
CJs retain their existing failure behavior because replacement cannot restore
their in-memory child-result receipts; automatic recovery is limited to leaf CJs.
The job's client registration must also remain valid: re-registering CP creates a
new token that the existing server job does not adopt. Migrating that registration
is outside this client-only recovery feature.

## One setting

`max_cj_restart_failures` in the client's `local/application.conf`, default `3`.
`0` disables recovery. No application opt-in is required.

The limit counts consecutive attempts to get a replacement initialized. There is
no lifetime restart limit. CP increments the counter when it launches a replacement
and resets it when that CJ completes runner initialization and reports READY.
The original failure does not consume an attempt. Three replacement startups that
fail consecutively end recovery; a successful replacement restores the full budget.
For a resident Slurm job (`execution_lifetime="job"`), a long-running assignment
can therefore make progress across repeated four-hour CJ allocations without a
lifetime restart-count limit, provided the application restores durable partial
checkpoints and the server's assignment/deadlines remain valid. For example,
two days of useful training may require twelve or more allocations once queueing,
startup and repeated work are accounted for. This is resident CJ replacement,
not automatic restart of an independently allocated task worker.

This measures initialization success, not training progress. A CJ that repeatedly
initializes successfully and then crashes can continue restarting. Application
checkpoint compatibility and progress remain the application's responsibility.

## Allocation wall-time and task lifetime

Replacing a failed **CJ** and restarting a separately allocated **task worker**
are different operations. `max_cj_restart_failures` controls the former; it is
not a retry budget for native task workers.

The [Slurm task-adapter proposal](https://github.com/NVIDIA/NVFlare/pull/5513)
is distinct from resident Slurm recovery. Its task-mode CJ does not opt into
restart, because independent task allocations may outlive it; confirming only
CJ allocation exit is insufficient. That proposal is not delivered support for
native task restart/resume.

For example, a two-hour allocation cannot complete a six-hour assignment merely
because `execution_lifetime="task"` is selected. The
[task checkpoint handoff](task_worker_lifecycle.md#application-checkpoint-state)
commits an application-selected candidate with a valid completed attempt. It
preserves state between successful assignments, not arbitrary progress from an
unfinished one. Periodically writing a file is not enough to make an interrupted
candidate committed, compatible with redelivery, or safe to adopt.

### Wall-limit and owner-loss timeline

For native task execution, CP owns scheduler execution and cleanup, CJ owns local
checkpoint promotion and result submission, and SJ owns assignment validity and
attempt identity. The worker produces candidate files; writing them does not
transfer checkpoint or result authority to the worker. This describes Kubernetes
and the Slurm draft's intended ownership, not a deployed Slurm qualification.

The successful order is **run -> seal completion -> confirm settlement -> validate
and promote checkpoint -> submit saved result -> receive terminal receipt**.
Compute can be released before receipt; payload cleanup has additional conditions.
Failures at those boundaries have different outcomes:

| Boundary/event | Execution and cleanup owner | Checkpoint/result outcome | Assignment and restart outcome |
| --- | --- | --- | --- |
| Worker allocation expires before a valid completion | Live CJ requests cancellation/settlement through the CP-owned native handle. CP retains unresolved ownership. | No new checkpoint or successful worker output. After a settled failure, surviving CJ can submit an execution-error reply through the normal result path. Retain the last committed checkpoint and failure evidence. | SJ still governs closure/deadlines and handles the error reply. No automatic physical task-worker retry or new attempt ID is created. |
| Worker has valid completion and confirmed settlement; result reply is lost | Worker no longer owns compute; surviving CJ owns saved-result submission. | A locally committed checkpoint remains committed. CJ resends the same saved output, not unfinished computation. | Retry preserves the existing identity; receipt handling prevents duplicate result effects. No new worker is launched. |
| CJ dies while a task allocation may still exist | Live CP retains cancellation/reconciliation ownership; CJ death alone does not establish worker settlement. | Durable files may survive, but no replacement CJ automatically adopts them or promotes an unfinished candidate. | Kubernetes and the Slurm draft disable task-CJ restart. SJ retains assignment authority; it does not promise to keep the assignment open. |
| CP dies or its own allocation expires | There is no supported restart/adoption of CP's task ownership. Native execution may outlive CP; operator reconciliation is required. | Durable checkpoints/results are files, not a reconstructed submission owner or proof that the worker stopped. | CP restart must not be treated as permission to relaunch old work. Automatic native recovery is not implemented. |
| Resident Slurm CJ allocation expires (`job` lifetime) | Surviving CP may replace CJ only after confirmed allocation exit and all recovery gates pass. | Application code restores its own compatible partial checkpoint; progress since that checkpoint may repeat. | Normal GET_TASK redelivery retains the outstanding task/attempt IDs. This existing resident path is distinct from launching a new native task-worker attempt. |

Thus the resident same-ID redelivery path is not a specification for native task
resume. A future native retry must coordinate a new physical attempt identity
with SJ and prove old execution has stopped before replacement. Saved-result
retransmission, by contrast, must not create another physical attempt.

### Native task-resume follow-up requirements

The following are requirements for a future design and qualification, **not
implemented restart behavior or a new configuration option**:

- Define a durable checkpoint publication and restore contract for partial
  progress within an assignment, including model, optimizer, data-loader and RNG
  state as required by the application. Required state must survive allocation
  loss without relying on shutdown upload or finalizers running.
- Retain ownership of every accepted or ambiguously submitted allocation. Prove
  old compute has settled and fence stale output before authorizing replacement;
  timeout expiry, a cancel acknowledgement or a missing scheduler record alone
  is insufficient.
- Coordinate with the server's assignment/closure state. A new task-worker
  execution needs a new server-issued physical attempt identity; retransmitting
  an already completed result does not. Specify checkpoint compatibility and
  replay policy without equating complete receipt with algorithm acceptance.
- Define who reconstructs CJ/worker ownership after coordinator loss, bounded
  retry and queue/deadline policy, cancellation during replacement, and behavior
  when the scheduler or shared storage cannot establish a safe state.

The future design must meet these observable correctness criteria; these are
behavioral requirements, not a runnable QA plan or claims of current support:

| Boundary | Required outcome for future resume support |
| --- | --- |
| Allocation expires during training, before a completed result | Restore only a verified compatible checkpoint and make progress across multiple allocations without concurrent old/new compute. |
| Allocation ends during checkpoint publication | Reject torn/uncommitted state and retain the last valid checkpoint. |
| Worker has completed but upload/receipt is delayed | Deliver the saved result without recompute or duplicate aggregation effects; compute is already released. |
| CJ is lost, cancellation races relaunch, or scheduler queries fail | Preserve ownership and diagnostics; do not replace work until settlement and assignment authority are established. |

Allocation time limits, application timeouts and queue timeouts are different
events. Resource release during blocked result upload and recovery of unfinished
training from a durable checkpoint are also separate claims: normal completion
or saved-result retransmission cannot establish mid-assignment resume support.

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

Normal success, STOPPED teardown, cancellation, configuration errors, and unsafe
component failures do not trigger recovery. Missing accounting, node loss, backend
API errors, and network/heartbeat failures are not proof of termination. A launch
exception ends recovery rather than risking duplicate submission when acceptance
is unknown. If a handle was returned before launch setup failed, CP terminates and
waits for that worker before releasing its resources. Docker physical launches use
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

### Process task-mode recovery gap

In the current Process task implementation, each task worker starts a separate
process group. `ProcessHandle.can_restart()` checks the CJ's own group and
`client_api_bootstrap_*.json` external-trainer files; it does not check task-worker
groups or exclude task lifetime. Consequently, CP can find CJ replacement eligible
without proof that its task workers have stopped. The worker's startup parent-PID
check is not CP-owned settlement evidence after a later CJ failure.

This is an implementation eligibility gap, not supported task-mode recovery and
not a guarantee that every replacement will launch overlapping compute. The
[local task-backend removal proposal](https://github.com/NVIDIA/NVFlare/pull/5592)
defers Process/Docker task mode instead of adding a new local recovery owner.
Until it lands, do not qualify Process task-CJ replacement as safe solely from
CJ-group exit. This documentation does not change or repair that restart gate.
See the [backend/lifetime matrix](task_worker_lifecycle.md#backend-and-recovery-boundaries).

## Task and checkpoint boundaries

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
