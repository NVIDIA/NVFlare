# Backend shutdown and cleanup contract

This describes PR [#5561](https://github.com/NVIDIA/NVFlare/pull/5561), including
its limits. It introduces no new state enum, status API, or partial-release policy.

## One rule

Keep an accepted or possibly accepted execution owned until backend settlement;
requesting cancellation, losing an observation, or seeing a leader exit cannot by
itself authorize parent cleanup. Parents use `wait()`, not a terminal `poll()`
return, as their release barrier. Physical shutdown is the safety target; the
legacy Slurm exceptions below mean this is not yet a universal proof guarantee.

Four facts remain separate: stop requested, application outcome known, execution
stopped, and cleanup performed. An unsuccessful launch can still own an execution.

## Backend evidence

| Backend | What the handle waits for | What is insufficient |
| --- | --- | --- |
| Process | Leader exit and quiescence of its launcher-owned process group, after success or failure. Linux can establish a zombie-only group using two complete process/thread observations. | Leader exit, one incomplete `/proc` scan, or an expired cleanup grace period. |
| Docker | Native exit/removal observation for the container ID. While terminating, `wait()` retries cancellation that failed to reach Docker. | A cancellation request or an inspection/transport error. Main already retained ownership in several Docker failure paths; this also fixes stalled cancellation. |
| Kubernetes | UID-bound Pod/container/sandbox evidence, finalizer reconciliation, and actual Pod removal on cancellation; details below. | DELETE acknowledgement, a bare 404, node-loss status, or a terminal phase alone. |
| Slurm | Retain uncertain submissions and files; recover a missing job ID with a unique launch name and validated ownership. For restartable terminal states, require accounting plus absence from the active queue. | A lost `sbatch` reply, an absent unknown-ID lookup, `scancel` acknowledgement, or an unavailable scheduler. Legacy terminal fallbacks remain below. |

Confirmed rejection or proven non-delivery releases unused launch assets. Ambiguous
creates/submissions retain a handle. Cancellation/observation retries reconcile
that attempt; they do not launch another execution.

## Kubernetes settlement details

Reads bind the returned Pod UID; ambiguous creates first require the matching
launch label. A different UID is not adopted or deleted as the old execution.
For a bound Pod, shutdown evidence requires:

- Terminal Pod phase, excluding `NodeLost`, `Shutdown`, and PodGC deletion evidence.
- `PodReadyToStartContainers=False`, establishing sandbox teardown.
- Stopped evidence for regular, init, and ephemeral containers. A failed regular
  init in a `Never` Pod can establish that later containers never started; generic
  waiting/missing status is insufficient.

An unbound Pod marked for deletion can settle as never scheduled. The handle
removes its settlement finalizer with UID/resource-version checks only after
accepting shutdown evidence. Normal completion need not delete the completed Pod;
cancellation waits for actual disappearance after finalizer handling.

A 404 settles only after recorded shutdown proof, **or** when all of these hold:
the handle bound the Pod UID, observed its own finalizer on that UID, and either
successfully sent deletion or observed it already deleting with the finalizer.
Disappearance in that second case relies on another controller/operator clearing
the finalizer. The operator runbook requires independent runtime shutdown
verification; arbitrary force removal is not physical proof.

Read/patch/delete cleanup requests use the short timeout. Pod create preserves
normal admission timing. Custom Roles need Pod `patch`; missing sandbox evidence
or failed finalizer patching can retain ownership indefinitely.

## Parent behavior and cleanup order

Client initial/replacement launches retain published handles through exceptions,
missing return handles, and post-launch failures. Cleanup requests cancellation
and waits while registration and required resources remain owned. A replacement
uses the separate `can_restart()` and CJ retry-policy gates.

Server abort captures its handle. The grace wait and first cancellation attempt
run in a non-daemon worker joined by server close before process-monitor teardown.
Only subsequent unbounded settlement is a daemon; it removes only the matching
registration. A successful first call is not guaranteed if the backend is down.

Kubernetes invokes its resource-release helper before returning from normal
settlement: it calls task-owner release, removes workspace-transfer registration,
and attempts Secret deletion. Task-owner release closes admission and requests
background worker cleanup; it does **not** join that thread. The independent CP
owner remains until workers settle, and CJ replacement additionally checks that
owner's thread has finished. This document does not add early compute release
ahead of the existing cleanup sequence.

## Known limits

- **Slurm:** five healthy accounting misses for a known ID, and terminal states
  such as `NODE_FAIL`/`CANCELLED`, can finish without `exit_confirmed`. This policy
  predates #5561; it contradicts a strict reading of the stronger shared `wait()`
  docstring. These paths do not establish restart safety. Changing them is an
  explicit backend-policy decision, not something a new enum would fix.
- **Process:** `poll()` can report the leader outcome before its group settles;
  both parent monitors call `wait()` before normal release.
- **Cleanup:** Kubernetes Secret deletion is best effort and forgets the name
  after logging a failure. Slurm artifact deletion can similarly log and continue.
  A returned handle does not promise every backend artifact was deleted.
- **Parent loss:** ownership adoption after CP/SP restart is unsupported. The
  pre-existing server initial-launch publication window also remains outside this
  change. Retained files alone do not establish a replacement's safety.
- **Qualification:** native API outages, node loss, operator reconciliation, and
  accepted-but-unacknowledged submissions need live backend validation.

## Slurm task-execution priority

The existing [task contract](../../nvflare/apis/task_launcher_spec.py) already has
`QUEUED`, `RUNNING`, `TERMINAL`, plus independent `settled`, `cancel_requested`,
and `failure_reason` fields. The [Slurm task adapter](https://github.com/NVIDIA/NVFlare/pull/5513)
should implement that contract with CP-owned cancellation and cleanup, including
CJ loss and delayed submissions. It must not turn the resident launcher's legacy
terminal fallbacks into `settled=True`. A new generic lifecycle framework is not
a prerequisite. See the [task lifecycle](task_worker_lifecycle.md) and
[CJ recovery](client_job_recovery.md) for the separate result and retry contracts.
