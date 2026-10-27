# Task worker lifecycle and integration contract

P3 provides one stateless Process worker for an ordinary Executor. P4 adds a plain,
role-neutral TaskSupervisor that owns its execution and artifact lifetime, plus
the CPU Process client integration. The resident Client Job owns filters, analytics
publication, result submission and receipts; its TaskWorkerExecutor launches a fresh
compute worker for each assignment. Local completion, physical process settlement, and server receipt are
separate facts.

## Client configuration and script lifetime

Client configuration accepts `execution_lifetime: "job"` (the default) or `"task"`.
Task-lifetime exports use client `format_version: 3`, which older clients reject
instead of silently running with job lifetime. Job-lifetime exports keep version 2.
Job lifetime keeps the existing executors and Client API backends. Task lifetime
authorizes inert executor and component specifications in the Client Job and again
in the worker before importing application code. Compute instances are constructed
only in the worker. Task data and result filters always stay in the Client Job.

In task mode, each worker constructs its executor and the explicitly supported
compute dependency, if any. Client API scripts create their own models and
dataloaders. StatisticsExecutor brings only the generator named by its
`generator_id` into the worker. Its existing task handler initializes and finalizes
the generator; the worker does not supply a generic helper lifecycle. The generator
must initialize from local data on each assignment and cannot rely on memory from
an earlier statistics task or pre-run handshake. The dataframe and image statistics
examples expose `--execution_lifetime task` for this profile.

Receivers, widgets, privacy filters and their cleansers remain resident in CJ.
The statistics payload codecs are registered in CJ for its privacy filters and in
the worker by the existing statistics handler. DecomposerRegister also remains in
CJ and is constructed in the worker before input decoding when custom payload
codecs are configured. Client API framework codecs are registered from the declared
exchange formats in CJ before receiving federation payloads, and in the worker
after authorization but before reading its input file. This serialization setup is
independent of compute lifetime.
Task workers do not build nested components or place dependency graphs.
`component_dependencies` and `execution_scope` are rejected on worker specifications.
The existing `args.class_attributes` constructor wrapper is supported without
nested component construction. Ordinary constructor dictionaries marked
`config_type: "dict"` remain data.
Jobs cannot configure the framework's internal adapter, launcher, environment
policy or declared task state.

The profile permits in-process ClientAPIExecutor, StatisticsExecutor and NPTrainer.
Executors requiring federation messaging or other resident engine services use job
lifetime; unqualified executors fail validation before export or application imports.

`FedJob(execution_lifetime="task")` and `job.set_execution_lifetime(...)` expose
these controls at the Job API level. The unified and PyTorch FedAvg and PyTorch
FedProx recipes accept `execution_lifetime="task"` in their constructors. Existing
recipes can use `recipe.add_client_config({"execution_lifetime": "task"}, clients=...)`
to configure existing client apps. Export validates GPU counts and preserves
resource/launcher settings for trusted receiving-site admission; unrelated
job-lifetime GPU apps in the same deployment remain valid. Resource defaults and
receiving-site overrides use the normal launcher resource resolvers. The default
CPU Process runtime rejects GPU requests. An `@ALL` client app
cannot be combined with a site-specific lifetime; use named client apps for
per-site lifetimes. Temporary execution-parameter overrides apply only to declared
apps and restore their original parameters after use.

Both lifetimes use the same InProcessClientAPI with small job/task input and
result adapters. The task ClientAPIExecutor supports `execution_mode="in_process"`. Its script
runs on the worker's main thread, receives one assigned model, and hands off one
eager result through the existing Client API model, format, FULL/DIFF, metadata and
rank semantics. The same receive/send loop exits after that send. Script arguments
retain their boundaries, and supported secret placeholders resolve in the worker.
Sending serializes the result durably to disk before returning and retains only
its artifact reference in the worker. Script failure or finalizer failure still
prevents completion. Each log call travels live to the Client Job over the
worker's cell session (`task_worker_cellnet_session.md`), which fires it into the
job while the attempt is still owned; a record the Client Job rejects fails the
attempt, while a receiver that raises is logged as in job lifetime. The
worker captures these records directly through the Client API adapter; it has no
local analytics event handlers.
Worker log lines are replayed through CJ logging after settlement, making them
available to the existing JobLogStreamer. Replay accepts only regular files, reads
at most 1 MiB and emits at most 4,096 lines; truncated logs remain in the workspace.
Worker failures include their original
error type, message and diagnostic path in the CJ exception.

Script process memory, optimizer state and executor instances do not survive the
assignment. Applications needing previous local model state must use a job-lifetime
script or manage their own explicit persistence; this profile supplies no state
declaration or aggregation admission mechanism. An opt-in opaque checkpoint file
can preserve application-selected state without preserving process memory.

For Client API scripts, `flare.system_info()["task_checkpoint"]` supplies `input`
and `output` paths. `input` is `None` on the first assignment, otherwise an isolated,
checksum-verified copy of the last committed checkpoint for this client job,
executor configuration and task name. `output` is a pending file unique to this
physical attempt. The script restores and saves it using its own framework:

```python
checkpoint = flare.system_info()["task_checkpoint"]
if checkpoint["input"] is not None:
    state = torch.load(checkpoint["input"], weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
# Train, then save any application-selected state.
torch.save({"optimizer": optimizer.state_dict()}, checkpoint["output"])
flare.send(output_model)
```

The worker seals the optional file in place after executor finalization. One immutable
completion record binds its checksum to the result. After physical settlement and
result validation, CJ atomically promotes the checkpoint together with its owning
completion identity and result reference. Atomic replacement is the local commit
point: if a later directory sync or temporary-file cleanup fails, CJ retains the
published checkpoint and returns its matching successful result, reporting a
warning that crash durability could not be confirmed. Failures before replacement,
cancelled promotion and malformed candidates leave the previous state unchanged.
A missing checkpoint also leaves it unchanged. Workers receive only an attempt-local
input copy, so accidentally overwriting it cannot change the committed checkpoint. Promotion is
idempotent for the same completion. Result-submission retries reuse the saved
result in CJ and never rerun training or promote state again. There is no automatic
physical retry or cross-CJ crash recovery; a new physical attempt requires a new
server-issued attempt ID. Promotion records local completion, not server acceptance
or aggregation admission. Successful terminal receipts release the attempt's
checkpoint files along with its other payloads. The committed checkpoint uses a
small atomic manifest and an immutable payload linked from the settled attempt,
so sealing and promotion do not copy the full file. The next worker receives one
isolated input copy. Superseded committed payload links are removed; one committed
checkpoint remains per client executor/task slot for the job. Retained attempts
share storage with their committed payload instead of duplicating it.

When the server resends the same assignment and attempt ID, CJ re-reads the
checksum-verified saved result. It does not rerun training, resend metrics or
promote that attempt's checkpoint again, including after a newer task completed.
Only attempts that finished CJ result processing are eligible. Missing, corrupt,
unsettled or already released results are rejected rather than re-executed.

The same hello-pt `client.py` runs in both lifetimes. It retains state in resident
job mode and explicitly restores optimizer momentum, the advancing DataLoader
generator and CPU PyTorch RNG state in fresh task workers, then writes a candidate
after training. It accepts earlier successful checkpoints after missed rounds and
starts fresh when a site has no checkpoint yet. CPU thread limits are opt-in. Its
paired numerical regression compares per-site incoming models,
returned updates and metrics in every round, plus global checkpoints. This qualifies
the checkpoint-aware example under matching CPU settings; changing only lifetime
does not make arbitrary unchanged stateful training code equivalent. Applications
must select and restore their own relevant state. Final global evaluation does not
write a checkpoint. Cross-site local-model submission remains a job-lifetime option.

The export validator checks P4's known unsupported workflow configurations using
actual workflow types, including subclasses. Both SCAFFOLD implementations are
rejected because their current clients do not restore control variates between
workers. Both cross-site evaluation implementations are rejected when they request
local-model submission, which the current task integration does not implement.
Global-only CrossSiteModelEval remains supported. Controllers have no task-lifetime
markers or validation hooks. Persistent state alone does not require job lifetime;
applications can explicitly checkpoint the state they need.

The default profile supports CPU Process workers on Linux and on macOS with Python
3.13+ providing waitid/WNOWAIT. The opt-in
[Process GPU profile](process_task_gpu_profile.md) adds static NVIDIA GPU
visibility on provisioned Linux sites, without local GPU scheduling or reservation. External/attach Client API sessions,
remote task launchers, and moving task orchestration to the Client Process are not
part of this profile. No server capability negotiation is added.
The Collab API's resident RPC service continues to use job lifetime.
Client API task workers also reject local-model submission and both SCAFFOLD wire
layouts at execution time, covering manually authored configs that bypass export.
Lightning `patch(..., restore_state=True)` rejects task lifetime because its
optimizer/scheduler cache is in-memory; use job lifetime or disable that callback
and restore native framework checkpoints explicitly.

## Site policy

Only the site's canonical `local/resources` configuration, including JSON,
CONF/YAML and default variants, supplies `task_execution`. Job application and
job resource overrides cannot select it.
The default launcher is ProcessTaskLauncher, environment forwarding is explicit,
and successful payloads are released after terminal receipt. An optional site configuration is:

```json
{
  "task_execution": {
    "environment_variables": ["TRAINING_SECRET"],
    "worker_timeout": 600,
    "settlement_timeout": 10,
    "poll_interval": 0.05,
    "finalization_timeout": 30,
    "artifact_cleanup": "receipt"
  }
}
```

`artifact_cleanup: "retain"` keeps every attempt. `max_payload_bytes` optionally
limits input and output payloads with the same policy. The job's executor definition
can shorten `worker_timeout`, and ClientAPIExecutor's `result_wait_timeout` bounds
application execution through script exit and finalizers. Framework result
serialization pauses this application deadline. After the result has been staged,
the site also imposes a finite `finalization_timeout` (30 seconds by default),
including when result_wait_timeout is unset. Once application finalization finishes,
checkpoint sealing and completion publication run outside these
application deadlines. The absolute `worker_timeout`, when configured, still bounds
the entire worker lifetime, including artifact I/O. Sending does not waive physical
settlement.

Workers receive only the framework's standard runtime environment allow-list plus
site-approved variable names. Federation credentials, Client API bootstrap and GPU
visibility cannot be forwarded; the default disables GPU visibility. The opt-in
Process GPU launcher applies its site-configured device IDs to worker visibility. Process workers share
the site UID and filesystem, so this is process lifetime isolation, not a security
sandbox. The site policy is excluded from client registration metadata.

## Execution and diagnostics

The worker checks executor, statistics generator and decomposer specifications
against site component policy before application logging, custom imports,
decomposers or input decoding.
It initializes the job audit sink first, preserving allow-all and warn-mode
authorization records. Compute objects and codec registration use the existing class
loader; no separate worker component builder is needed.

The worker uses direct `initialize(fl_ctx)`, `execute(...)`, and `finalize(fl_ctx)`
calls on its executor or Client API runner. Finalization also runs after partial
initialization and execution failure. The first failure is retained in failure.json;
a later cleanup error cannot replace it. StatisticsExecutor propagates generator
finalization failures to its caller so they cannot produce a successful task
completion. Decomposer registration is constructor
setup, with no lifecycle calls. Lifecycle events, registered event handlers and
before/after-task event hooks are not dispatched in the worker.

For Client API scripts, one private task runner directly initializes the shared
InProcessClientAPI, stages input through TaskInProcessAdapter, executes the script
and finalizes its resources. It has no separate mode-specific backend or backend
interface wrapper. ClientAPIExecutor supplies validated settings
but its event lifecycle is not invoked. The shared event dispatcher and job-mode
backend selection require no task-worker hooks. Existing FLComponent logging is
written normally; its subsequent log-event notifications are ignored. The
system_panic compatibility call latches a failure without dispatching handlers;
other event requests fail explicitly. Filters, widgets, application events and
federation event routing remain owned by the CJ.

Wall-clock start/end timestamps are diagnostic and may move backward. Elapsed
time uses time.monotonic(). Launcher cancellation can terminate the worker without
running finalizers. Executors must support this stateless one-shot cancellation
contract; finalization is not guaranteed after process termination.

There is no per-task guardian process. CJ owns the worker process group through
its launcher, including normal cancellation, descendant settlement and shutdown
deadlines. Abrupt CJ death can leave an active worker or its descendants running.
Independent crash cleanup owned by CP is a follow-up; this profile does not claim
that guarantee.

## Payload storage

PyTorch task payloads use the [disk-backed payload format](disk_backed_task_payloads.md).
The GET_TASK route decodes pass-through references. Before task-data events, the CJ
resolves inputs eagerly if a site or job data filter applies, or to local tensor
files otherwise. On unfiltered tasks, CJ input hooks receive local lazy refs with
metadata and `nbytes`; tensor operations such as `.shape`, `.numel()` and arithmetic
are unavailable. Workers always receive ordinary mapped tensors. The existing
clearing of `TASK_DATA` after execution, before result filters, is unchanged.

Input and result artifacts contain small FOBS records with tensor file descriptors
and hashed per-tensor safetensors files alongside them. Received single-tensor files
are hard-linked into the attempt on the same filesystem; other tensors are written
one at a time. The runner releases receive scratch files in a `finally`, including
when execution never starts. Result callbacks and the sender receive mapped tensors,
so callback edits are transmitted and already-open mappings survive filename
cleanup. Retained result files are revalidated and mapped for resubmission. Many
small tensors require many files; temporary serialization remains proportional to
the largest tensor plus the existing sender's bounded cache and prefetch.

There is no fixed 4 GiB payload ceiling. FileTaskArtifactStore and WorkerBootstrap
accept an optional positive max_payload_bytes supplied by the trusted site/client
adapter. The adapter must apply the same policy to input staging, the bootstrap,
and result reading. None leaves capacity to the filesystem and its quota. The
separate bootstrap and metadata record limits remain in place. The payload limit
counts the FOBS record and all referenced tensor files together.

After confirming writer and reader quiescence, release_payloads also removes
interrupted publication temporaries for its payload/metadata names. Full retention
cleanup can remove an incomplete attempt directory whose identity.json was never
published. An existing identity record must still be valid and match the caller.

FOBS serialization hashes each byte as it is written. The worker still verifies
the stored result before publishing completion, and the reader verifies it before
decoding. These verification passes detect changed or corrupt payloads; staging
no longer rereads the complete file merely to calculate its initial digest.

Trusted bootstrap directory aliases, including /tmp on macOS, are resolved before
opening. Bootstrap files themselves and attempt artifacts cannot be symlinks.
Directory ancestors require search permission; directories receiving durable
artifact writes also require read/write permission for the durability barrier.

## Completion and supervisor integration

The worker serializes completion publication with fatal, abort, and direct failure
reporting. A failure observed before the publication decision prevents completion.
A direct compute or analytics operation still in progress also prevents it.
Successful publication closes the runtime to subsequent operations and abort requests.

A valid completion.json can coexist with a later nonzero process exit, such as
SIGKILL after publication, or with failure.json if the directory fsync fails after
the completion link is installed. The worker preserves both records for diagnosis.
A directory durability error means persistence across a host crash is uncertain.

P4 must validate completion identity, worker PID, payload size and digest, and
confirm physical settlement before consuming a result. A fully verified completion
is evidence of finished local computation even when a later process error or
post-publication failure diagnostic exists. Those later facts do not erase local
completion. TaskSupervisor returns the verified result and the observed execution
status independently, retaining payloads when execution failed or failure.json
exists. Missing or invalid completion still fails validation after settlement.

The supervisor has one job lifetime and admits one active physical attempt at a
time. Unconfirmed settlement permanently prevents further attempts on that
instance and retains its handle and artifacts. end_run closes admission and
cancels known execution. If launch is still acquiring a handle, the caller must
join its run call before discarding the supervisor; that call owns and cancels
the late handle. There is no reset or automatic physical retry.

For publication, the caller validates the reply's workflow, task, and attempt
before calling record_receipt. RETRY or unknown receipt retains the saved result.
Default cleanup requires confirmed settlement, a decoded successful result and
RECEIVED or TASK_CLOSED. Uploads use the decoded result snapshot. Saved-result replays
pin disk payloads until their readers finish, independently of compute admission.
Cleanup is independent of admission of the next assignment.
Explicit retention, failed cleanup and failed attempts keep their evidence. Completion and diagnostic
records remain after successful payload release. Closure says nothing about an
earlier upload's receipt or aggregation use.

The shared worker entry requires an explicit owning parent PID for Process
execution. Native Docker/Kubernetes/Slurm workers select native execution scope
and do not rely on a local CJ parent or Process PID/group assumptions.

Native task cleanup holds `END_RUN` until the supervisor no longer owns physical
execution and the executor call has finished, including late launch handles.
Bounded cleanup failures are retried after backend recovery. A readiness timeout
is not settlement; the settlement-unconfirmed fence remains sticky after recovery.

A local completion never overrides Client Job/task cancellation or authorizes
network publication by itself. The adapter must drain pending launches during job
shutdown, recheck cancellation before sending, retain the saved result through
receipt retries, and release successful disk payloads after terminal receipt.
Sim/POC/Prod E2E tests must exercise these boundaries, including process
termination and filesystem failure after completion publication.

Released attempts leave the supervisor cleanup set and drop their owned handles.
Lightweight launch identities remain for the job lifetime to prevent relaunch.
