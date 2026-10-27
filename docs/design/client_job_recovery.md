# Automatic client-job recovery

CP automatically replaces a failed CJ using the same job ID, launcher, metadata,
resource reservation, and persistent job workspace. The replacement uses normal
`GET_TASK` delivery: the unchanged server resends an outstanding assignment with
its existing task and attempt IDs. User code restores its own checkpoint.

Implemented for Process, Docker, and Slurm. Kubernetes is deferred because its
current CJ workspace is ephemeral. CP and the server must remain running.
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
Thus a two-day task can span twelve four-hour allocations without exhausting it.

This measures initialization success, not training progress. A CJ that repeatedly
initializes successfully and then crashes can continue restarting. Application
checkpoint compatibility and progress remain the application's responsibility.

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

Process CJs that have created a disposable task runtime are not automatically
restarted. Task workers own separate process groups, so CJ group exit alone cannot
prove their settlement. The existing `.nvflare/task-execution` directory retains
attempt evidence and conservatively blocks replacement, even if earlier attempts
settled. Operator cleanup is required after abrupt CJ death. Ordinary task failure
or OOM can still be followed by a fresh task within the living CJ after confirmed
worker settlement. No additional restart-state file or allocation ledger is used.

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
