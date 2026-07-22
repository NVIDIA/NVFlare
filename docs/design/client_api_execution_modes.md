# Client API Execution Modes

## Status

**Implemented architecture, updated 2026-07-14.**

This document records the architecture implemented for `ClientAPIExecutor`.

The currently available modes are:

| Mode | Backend | Trainer location | Availability |
|---|---|---|---|
| `in_process` | DataBus | Thread in the Client Job (CJ) process | Available |
| `external_process` | Cell | Process tree launched and owned by the CJ | Available |
| `attach` | — | — | Reserved; not implemented |

Selecting `attach` fails clearly rather than silently falling back to another transport.

## Goals and Boundaries

The trainer-facing API remains:

```python
import nvflare.client as flare

flare.init()
model = flare.receive()
result = train(model)
flare.send(result)
flare.shutdown()
```

The implemented architecture has these boundaries:

- `ClientAPIExecutor` selects one internal backend and drives only its
  `initialize` / `execute` / `finalize` lifecycle.
- `InProcessBackend` uses DataBus and runs the training script in the CJ process.
- `ExternalProcessBackend` runs Client API jobs in a launched subprocess and communicates
  with the trainer directly over Cell.
- Cell, FOBS, `ViaDownloader`, and `DownloadService` provide serialization, large-object
  transfer, progress, and terminal transfer status. The Client API does not add another
  payload wrapper or streaming protocol.
- Framework representation conversion is declared in `TASK_EXCHANGE` and performed at the
  trainer-side Client API receive/send boundary, not by the executor or Cell protocol.
- Job runtime placement remains the responsibility of `JobLauncherSpec`. Training-process
  launch inside the CJ is a separate lifecycle owned by `ExternalProcessBackend`.

## Execution Mode Selection

`ScriptRunner` constructs `ClientAPIExecutor` by default. The existing
`launch_external_process` flag selects the backend:

```python
ScriptRunner(script="train.py")  # ClientAPIExecutor(in_process)

ScriptRunner(
    script="train.py",
    launch_external_process=True,  # ClientAPIExecutor(external_process)
    command="python3 -u",
)
```

`execution_mode` remains available as an explicit mode override and for future modes such as
attach.

For `external_process`, the resulting path is:

```text
ClientAPIExecutor
  -> ExternalProcessBackend (CJ)
  -> Cell request/reply + FOBS/ViaDownloader
  -> CellClientAPI (trainer)
```

## External Process Architecture

### Actors

- **CJ:** hosts `ClientAPIExecutor` and `ExternalProcessBackend`.
- **Trainer process tree:** runs the user's script or a command such as `torchrun`. The backend
  launches and owns this local process tree.
- **Control rank:** rank 0 is the Cell peer and calls the Client API. Other distributed ranks use
  their framework's own collectives.

Only one `ClientAPIExecutor` is configured for a client job. It routes all of its configured task
names through the selected backend.

This is a lifecycle/control-plane constraint, not an assumption that task routing alone serializes
the system. Every configured executor receives `START_RUN` before task routing begins; multiple
Client API executors would therefore initialize concurrently and collide on the in-process DataBus
binding or the shared CJ Cell protocol callbacks. Job construction and raw client-config loading
reject that configuration early instead of carrying a second runtime ownership guard.

### Launch and typed bootstrap

Before launching a trainer, `ExternalProcessBackend` writes a launch-scoped JSON bootstrap file.
The file is a typed rendezvous envelope, not a new transport or data plane. It contains the
metadata needed before a Cell session exists:

- bootstrap schema version and execution mode;
- Cell connect URL, CJ FQCN, and prescribed trainer FQCN;
- launch-scoped binding token;
- job and site identity;
- task-exchange and memory-management configuration needed by the trainer-side API.

It contains no task model, result payload, payload manifest, or transfer state.

The writer creates an owner-only (`0600`) sibling temporary file and atomically installs it with
`os.replace`. Each launch gets a fresh filename, FQCN, and token. The backend passes the file path
through `NVFLARE_CLIENT_API_BOOTSTRAP`; an explicit `flare.init(config_file=...)` can also select
the Cell Client API from the typed `schema_version` / `execution_mode` envelope.

The bootstrap exists because the trainer must know how to create its Cell and reach the CJ before
the first Cell message can be exchanged. Once the Cell session is established, all task, result,
log, heartbeat, abort, and shutdown traffic uses Cell. The bootstrap file is not polled and is not
used for ongoing communication.

### Session and control protocol

The launched trainer reads the bootstrap, creates `CellClientAPI`, and sends `HELLO`. The backend
validates the launch identity, rank, protocol version, origin FQCN, job/site scope, and current
launch token before returning `HELLO_ACCEPTED` with the session and heartbeat policy.

V1 assumes a trusted host for launch availability. A same-host process that can claim the
prescribed trainer FQCN can race the real trainer with a bogus token and cause that launch to fail.
The process cannot authenticate the session or access task/result data; this is a bounded launch
denial-of-service risk, not an authentication or data-disclosure bypass.

The implemented per-task exchange is:

```text
CJ -> trainer : TASK_READY(task_id, task_name, Shareable)
trainer -> CJ : TASK_ACCEPTED | TASK_FAILED

trainer -> CJ : RESULT_READY(task_id, Shareable)
CJ -> trainer : RESULT_ACCEPTED | RESULT_REJECTED
```

`TASK_ACCEPTED`/`TASK_FAILED` and `RESULT_ACCEPTED`/`RESULT_REJECTED` are Cell request replies.
LOG and HEARTBEAT use the same Cell connection. ABORT and SHUTDOWN provide task/session teardown.
The task ID correlates each single task/result delivery attempt.
There is deliberately no receiver-only dedup cache while no sender retries exist. Attach-mode
redelivery tolerance must add sender retry and receiver deduplication together.

There is no separate Client-API payload envelope, manifest, transfer ID, or payload-transfer
state machine layered over these messages.

### Payload handling

The task/result `Shareable` is placed directly in the Cell request. Cell encodes the request with
FOBS. Small objects remain inline; large supported objects are represented by the existing
`ViaDownloader` decomposer and served through `DownloadService`.

```text
Shareable in Cell request
  -> FOBS encoding
  -> inline data or ViaDownloader reference
  -> Cell/DownloadService materialization at the consuming receiver
  -> request handler
```

The Client API supplies call-scoped FOBS context only to observe the actual transactions created
by that encode operation. It waits on those `DownloadService` transaction outcomes and uses
Cell's progress-aware request wait. It does not copy bytes, create a second reference format, or
maintain a parallel payload registry.

Task direction:

1. The CJ sends `TASK_READY` with the task `Shareable`.
2. Cell/FOBS materializes the request in the trainer process.
3. Only then does the trainer's `TASK_READY` handler validate and queue the task and return
   `TASK_ACCEPTED`.

Result direction:

1. The trainer sends `RESULT_READY` with per-message Cell pass-through enabled and declares the
   receiver identities supplied with the task. Those are the ultimate server/workflow receivers
   when the workflow supplies them.
2. Cell/FOBS invokes the CJ handler with inline values and/or lazy `ViaDownloader` references.
3. The CJ validates and stores that result, then returns `RESULT_ACCEPTED`; this acknowledges the
   envelope, not completion of every downstream tensor download.
4. `ClientRunner` passes the transport-delivered result to events, filters, and the workflow
   without adding automatic materialization or receiver rewriting. When lazy references remain
   unchanged, the server/workflow downloads directly from the trainer.
5. The trainer waits for the strict terminal outcome of every created `DownloadService`
   transaction before releasing its result resources or allowing a one-task process to exit.
   The last accepted receiver confirmation settles a completed transaction immediately; the
   waiter does not depend on the periodic expiration monitor noticing it. A receiver that does
   not provide terminal confirmation has no acknowledgement after its terminal serve, so that
   path remains monitor-settled: this preserves a post-reply interval before a one-shot producer
   can observe completion and tear down its Cell.

In the task direction at the CJ entry point, the CJ decodes only the
`(SERVER_COMMAND, GET_TASK)` route lazily for the external-process executor. `ClientRunner`
passes that representation through its event/filter/executor sequence without adding a
materialization policy. Other topics on `SERVER_COMMAND` keep ordinary decoding. The trainer Cell
always materializes the task before its Client API handler runs.

### Process and session lifecycle

`launch_once=True` keeps one trainer process/session for the job. `launch_once=False` creates a
fresh launch-scoped bootstrap and process for each task. The existing `TASK_EXCHANGE.launch_once`
lifecycle bit is included in the bootstrap config. After the CJ accepts a per-task result envelope,
the trainer remains alive until its complete result-publication barrier settles: the
`RESULT_ACCEPTED` reply reaches `send()` and every actual receiver download, when present, reaches a
terminal outcome. It then closes its Cell synchronously, and the CJ reaps that natural process exit
asynchronously. An orderly job SHUTDOWN cancels an incoming task materialization but not an already
accepted result publication. If teardown finds such a publication still active, it asks the trainer
to stop, starts the natural-exit reaper, and holds END_RUN until that truthful terminal exit. This
also covers inline results, whose acceptance reply can race END_RUN even though they create no
download transaction. Teardown cannot safely return with a daemon reaper because ClientRunner
tears down streaming and the CJ Cell immediately after END_RUN. The lower `DownloadService`
idle/receiver policy normally bounds a stalled transfer. END_RUN also applies a final total wait
backstop just beyond the default streaming-idle budget; if a still-connected trainer remains wedged
past that bound, the backend force-stops its owned process tree rather than hanging job teardown.
Other failure/teardown paths use the bounded SHUTDOWN/TERM/KILL sequence immediately.

Startup waits at most `launch_timeout` for the trainer to complete its HELLO handshake. The
default is 300 seconds; callers may explicitly use `None` when an unbounded wait is required. For
ordinary shutdown, a `shutdown_timeout` of zero is kept as zero and starts process-tree termination
immediately after the orderly SHUTDOWN notification. An accepted result whose publication is
different: its truthful terminal barrier takes precedence over ordinary process-shutdown timing,
so historical `ScriptRunner` defaults cannot erase the source-lifetime contract.

The backend starts the command in an owned process group on POSIX, monitors process-group exit and
the authenticated heartbeat lease, and rejects messages from stale sessions or unexpected Cell
origins. With a positive orderly-shutdown bound, shutdown sends a bounded Cell request and charges
its acknowledgement time against that bound; a zero bound keeps the immediate fire-and-forget
notification for ordinary teardown. A live accepted-result publication always uses a short,
acknowledged SHUTDOWN request because the trainer cannot be force-stopped before its send barrier
settles; the source reaper retries transient control-path failures. Its acknowledgement also
reports whether `send()` still owns that barrier. This state transition is serialized with
SHUTDOWN: either `send()` sees the stop and closes after terminal settlement, or the backend learns
that settlement already happened and can stop the persistent process. A standard trainer loop also releases its Cell
synchronously when `receive()` or `is_running()` observes the shutdown (or abort), so existing
scripts do not need an explicit `flare.shutdown()` at loop exit. The backend then terminates any
surviving owned tree with a bounded soft/hard stop sequence. On Windows the current implementation
uses `taskkill /T` for tree termination. Because a directly launched trainer does not run under
`MainProcessMonitor`, its Cell Client API shutdown also retires process-global DownloadService
state, the reliable retry scheduler, and the shared streaming executors; otherwise their non-daemon
pools can keep a completed one-shot or distributed worker process alive. That irreversible runtime
shutdown is specific to the dedicated Cell trainer process. In-process Client API contexts do not
retire those process-global services; a stopped context is evicted so a later job/session in the
same process can initialize a fresh one.

`flare.init()` also binds its returned context to the calling thread. Client API calls without an
explicit `ctx` prefer that binding; an old stopped binding is retained as a tombstone rather than
falling through to a newer process-global default. This prevents an abandoned in-process trainer
thread from publishing into, or shutting down, a successor job after the DataBus owner has closed
its API. A genuinely unbound helper thread retains the compatibility fallback to the current
process default; ownership-sensitive helper work should pass an explicit `ctx`.

A persistent (`launch_once=True`) script that returns or raises without re-entering
`receive()`/`is_running()` must call `flare.shutdown()` in a `finally` block. The same applies to a
per-task script that exits before its result is accepted; otherwise the backend's bounded forced
reaping remains the safety net.

The trainer also monitors CJ heartbeat/Cell availability. A session failure unblocks Client API
calls rather than leaving the trainer waiting indefinitely.

## Parameter Representation and FULL/DIFF

The execution-mode architecture does not use `ParamsConverter` or a ParamsConverter adapter.

`ScriptRunner` maps its framework setting to an explicit `params_exchange_format` and carries
that declaration, plus `server_expected_format`, through `ClientAPIExecutor` into
`TASK_EXCHANGE`. No framework is inferred from payload values and job construction does not
import PyTorch or TensorFlow.

The trainer-side Client API honors the declaration at its API boundary:

- receive: server representation -> framework-native `FLModel.params`;
- send: framework-native `FLModel.params` -> server representation.

The implementation is a small functional adapter with caller-owned state for PyTorch tensor
shapes and non-tensor entries; it does not recreate the `ParamsConverter` component
hierarchy. `RAW` explicitly means no representation adaptation. If either side of a declared pair
is `RAW`, conversion is a no-op and the payload passes unchanged; `RAW` is an adaptation off-switch,
not a concrete representation guarantee.

`params_transfer_type` is intentionally different from representation conversion. `FULL` versus
`DIFF` describes model state. It remains in the `ClientAPIExecutor` configuration, travels in the
trainer's task-exchange metadata, and is applied by the trainer-side Client API when preparing a
result. A DIFF is computed in the trainer-native representation before outgoing conversion.

CJ task-data/task-result filter ordering is unchanged. Filters receive the payload representation
delivered by the transport, which can contain lazy references when pass-through is active. This
execution-mode change does not make filter presence imply CJ materialization and does not add an
executor or filter capability contract.

Relocating content transformations from CJ filters to explicit send/receive endpoints is deferred
to a separate design and change. That work must first inventory the existing privacy, HE,
compression, selection, and blocking filters, including which side is trusted, which representation
each operation requires, and whether removing or blocking a lazy reference must explicitly abandon
its source transaction. This execution-mode change does not add a partial ``supports_lazy_payload``
filter contract.

## In-Process Mode

`InProcessBackend` retains the established DataBus behavior:

```text
ClientAPIExecutor
  -> InProcessBackend
  -> DataBus
  -> InProcessClientAPI in the trainer thread
```

There is no bootstrap file, Cell connection, or subprocess lifecycle. The same trainer-side
format declaration/adaptation and `params_transfer_type` model-state rule apply to this new
execution-mode path.

## Analytics

The trainer sends LOG messages through its selected Client API backend. `ClientAPIExecutor` owns
the conversion to NVFlare analytics events:

- in-process logs arrive through DataBus;
- external-process logs arrive over the session Cell.

The external-process backend delivers analytics without a separate metrics transport.

## Validation

Validate at least:

- existing subprocess Client API examples;
- Python and `torchrun` launch commands;
- FULL and DIFF training;
- PyTorch/TensorFlow native-to-NumPy conversion where configured;
- LOG/analytics delivery;
- large tensor streaming and tensor disk offload;
- abort, timeout, trainer exit, CJ loss, and process-tree cleanup;
- both `launch_once` policies.

## Deferred Work

- `attach` remains reserved and unimplemented. Its external ownership, credential delivery, and
  reconnect policy require a separate implemented contract.

## Implementation References

- `nvflare/app_common/executors/client_api_executor.py`
- `nvflare/app_common/executors/client_api/external_process_backend.py`
- `nvflare/app_common/executors/client_api/in_process_backend.py`
- `nvflare/client/cell/api.py`
- `nvflare/client/cell/bootstrap.py`
- `nvflare/client/cell/defs.py`
- `nvflare/client/converter_utils.py`
- `nvflare/job_config/script_runner.py`
- `nvflare/fuel/f3/cellnet/cell.py`
- `nvflare/fuel/utils/fobs/decomposers/via_downloader.py`
