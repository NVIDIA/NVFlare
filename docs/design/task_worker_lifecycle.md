# Task execution mode: architecture and lifecycle

Task execution mode keeps federation coordination alive while releasing compute
resources between assignments. A resident Client Job (CJ) receives tasks, runs
filters, publishes analytics and submits results. Its TaskWorkerExecutor launches
a fresh compute worker for each assignment. The Client Parent (CP) starts and
monitors CJs; the Server Job (SJ) runs the federated workflow. These roles exist
in both execution modes:

| Role | Job execution mode | Task execution mode |
| --- | --- | --- |
| CP | Starts and monitors each CJ. | Starts and monitors each CJ; owns native Kubernetes task execution. |
| CJ | Runs coordination and compute in one resident job runtime. | Keeps coordination resident and delegates each assignment's compute. |
| Task worker | No separate task runtime. | Constructs compute, executes one assignment, finalizes and exits. |
| SJ | Issues assignments and receives results. | Uses the same assignment and receipt protocol. |

Local completion, physical runtime settlement, result receipt and aggregation
acceptance are separate facts. This document defines the shared lifecycle;
the [resource and launcher metadata contract](job_meta_launcher_config.md),
[assignment and receipt contract](task_assignment_admission.md),
[disk-backed payload contract](disk_backed_task_payloads.md), and
[Kubernetes operator guide](../user_guide/admin_guide/deployment/kubernetes_task_launcher.rst)
describe their respective boundaries. Each document states its own assumptions
and can be read without an implementation-phase numbering scheme.

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
to configure existing client apps. Export validates resource shapes without
selecting a site's backend. CPU and GPU compute are both supported by the task
contract; the trusted receiving backend must enforce the requested resources or
reject them before compute. Kubernetes implements CPU and GPU workers. Availability
of other native backends depends on their installed implementation. Unrelated
job-lifetime apps keep their existing admission behavior. An `@ALL` client app
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
prevents completion. Analytics are serialized to a disk spool as each log call
arrives, then sealed and replayed by the Client
Job after settlement. The worker captures these records directly through the
Client API adapter; it has no local analytics event handlers.
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
checksum-verified saved result. It does not rerun training, replay analytics or
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

The export validator checks unsupported task workflow configurations using
actual workflow types, including subclasses. Both SCAFFOLD implementations are
rejected because their current clients do not restore control variates between
workers. Both cross-site evaluation implementations are rejected when they request
local-model submission, which the current task integration does not implement.
Global-only CrossSiteModelEval remains supported. Controllers have no task-lifetime
markers or validation hooks. Persistent state alone does not require job lifetime;
applications can explicitly checkpoint the state they need.

Process supervision requires Linux, or macOS with Python 3.13+ providing
waitid/WNOWAIT. Kubernetes uses the same worker entry point in native execution
scope and delegates physical supervision to CP. The current site profile requires
the same backend throughout CP, CJ and task workers. CPU versus GPU is a resource
choice within a capable backend, not a second execution mode. External/attach
Client API sessions remain outside this one-assignment worker contract. No
server capability negotiation is added.
The Collab API's resident RPC service continues to use job lifetime.
Client API task workers also reject local-model submission and both SCAFFOLD wire
layouts at execution time, covering manually authored configs that bypass export.
Lightning `patch(..., restore_state=True)` rejects task lifetime because its
optimizer/scheduler cache is in-memory; use job lifetime or disable that callback
and restore native framework checkpoints explicitly.

## Coordinator and compute resources

Job metadata separates the compute budget from the coordinator budget:

| Metadata | Job execution mode | Task execution mode |
| --- | --- | --- |
| `resource_spec` | Resources for the resident CJ, including its compute. | Resources for each task worker. |
| `cj_resource_spec` | Rejected for resident client apps. | Independent resources for the resident coordinator. |
| `launcher_spec` | Backend-specific runtime settings. | Backend-specific compute/runtime settings; does not become the CJ resource budget. |

For example, a Kubernetes task job can request two CPUs and one GPU per worker,
while its CJ requests 0.25 CPU, 512Mi and zero GPUs. The two budgets never inherit
from each other. Export/submission derives an empty CJ entry for every task site
when the author supplies no coordinator override; that entry selects site defaults
and keeps the compute GPU out of the whole-job reservation. CP and CJ validate
the locally resolved execution lifetime before constructing application components.
Submission reads client declarations without resolving unrelated client-only
environment variables or opening client HOCON includes. A lifetime declaration
must be explicit and literal for submission to derive task admission; locally
resolved configuration is still checked at runtime.

Kubernetes reserves a task worker's resources only while its Pod exists. Two CJs
at one site can therefore wait for and successively use one available GPU. There
is no NVFlare fairness or alternation guarantee: Kubernetes decides scheduling.
Task `queue_timeout` defaults to no resource-queue deadline; `startup_timeout`
defaults to 120 seconds and covers scheduling/startup stalls outside a confirmed
CPU/memory/GPU shortage. Configured absolute worker deadlines can still expire
while queued. Resident job admission keeps its existing resource reservation and
the K8s job launcher's separate `pending_timeout` behavior.

Backends must not silently discard licenses, GPU-memory quotas or custom resource
manager requirements when separating CJ admission from compute. Kubernetes and
the Process implementation reject compute requirements they cannot enforce.
Process validates unsupported compute requirements before starting a normalized
task CJ as well as before each execution. It has no implicit CPU/memory limiter
or resource allocator. This contract
does not add an NVFlare GPU scheduling ledger.

## Site policy

Only the site's canonical `local/resources.json` or its default fallback supplies
`task_execution`. Job application and job resource overrides cannot select it.
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
checkpoint sealing, analytics sealing and completion publication run outside these
application deadlines. The absolute `worker_timeout`, when configured, still bounds
the entire worker lifetime, including artifact I/O. Sending does not waive physical
settlement.

Workers receive only the framework's standard runtime environment allow-list plus
site-approved variable names. Federation credentials, Client API bootstrap and GPU
visibility cannot be forwarded as arbitrary job environment. CPU Process workers
mask GPU visibility; capable native GPU launchers arrange visibility through their
trusted resource binding. Process workers share the site UID and filesystem, so
this provides process lifetime isolation rather than a security sandbox. Native
worker environments come from their site's backend configuration. The site policy
is excluded from client registration metadata.

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

There is no per-task guardian process. For Process execution, CJ owns the worker
process group, normal cancellation, descendant settlement and shutdown deadlines.
The worker requires an explicit live parent PID greater than one and its own
process group; abrupt CJ death does not provide an independent CP cleanup guarantee.

For Kubernetes, the CJ is a tokenless proxy. Before starting it, CP binds a task
backend to trusted site policy and admitted job resources. CJ sends bounded JSON
launch/poll/cancel commands over CellNet with a per-job capability. That token
authenticates the request; a sender-set origin header is an additional consistency
check. Jobs cannot choose Pod manifests, namespace, service accounts or mount
authority. CP admits one unsettled attempt per job and retains cleanup after CJ
exits, including when create is still in flight or its reply is lost. Unknown
cancel identities have a 128-entry fence budget. The site-configured
`max_attempts` (default 10,000) also bounds all retained identities, including
cheap rejected launches. Exceeding either limit closes admission and cancels
owned execution without evicting earlier fences. The normal job handle releases task ownership
on CJ exit; a ten-second independent observer provides a backstop.

A Kubernetes delete response alone does not prove settlement. CP uses Pod UID
preconditions, a finalizer and shutdown evidence before releasing physical
ownership. Transient API observation errors preserve the last-known phase. Lost
shutdown evidence or ambiguous create outcomes keep the attempt fenced. CJ end-run
makes bounded cleanup calls and can exit while CP continues reconciliation. CP
restart adoption, automatic worker retry and automatic CJ resume are not provided.

## Kubernetes namespaces, storage and isolation

Kubernetes uses one configured jobs namespace for all CJs and their task Pods.
CP may run in that namespace or a separate control namespace with the required
cross-namespace permissions. A worker always uses its CJ's namespace; the launcher
does not create a namespace for each job.

The persistent profile uses a private CP workspace PVC plus a shared jobs PVC.
CP alone mounts the full jobs claim and creates a fresh hashed job/site directory;
CJ and task workers mount only that assigned subPath. Task execution requires this
durable profile. Resident jobs retain the existing emptyDir default; a site can
set `persist_resident_workspaces: true` on ClientK8sJobLauncher to put their
workspaces and checkpoints on the same jobs claim. Enabling persistence retains
job subtrees after successful completion as well as failures. The operator must
budget storage, archive and remove closed workspaces after all runtimes, in-flight
requests and readers are quiescent. No automatic subtree garbage collector is
provided. A definitively rejected Pod create releases only its newly created,
still-empty directory; an ambiguous create retains its storage fence.

CP, CJ and workers use the same configured nonroot UID/GID. Directory mode 0700
therefore does not isolate one job identity from another; cross-job filesystem
isolation relies on the kubelet subPath mount boundary and storage implementation.
Neither CJ nor worker receives a Kubernetes API token. Trusted templates cannot
add a second mount of the full jobs claim. Sites requiring distinct filesystem
identities or per-job claims need an additional isolation profile; this profile
does not allocate UIDs, chown workspaces or provision claims per job. Do not share
CP's private workspace claim with jobs. Application persistence on the jobs PVC
does not itself implement recovery after CJ or CP loss.

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

The supervisor validates completion identity, worker PID, payload size and digest, and
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
RECEIVED or TASK_CLOSED. A saved-result replay pins the attempt while it verifies
and decodes or maps its files; concurrent receipt cleanup waits for those readers.
The sender serializes the returned tensors, whose existing POSIX mappings survive
unlink. Additional consumers that reopen paths must arrange ownership before
cleanup; there is no general reader-lease API. Cleanup is independent of admission
of the next assignment.
Explicit retention, failed cleanup and failed attempts keep their evidence. Completion and diagnostic
records remain after successful payload release. Closure says nothing about an
earlier upload's receipt or aggregation use.

A local completion never overrides Client Job/task cancellation or authorizes
network publication by itself. The adapter must drain pending launches during job
shutdown, recheck cancellation before sending, retain the saved result through
receipt retries, and release successful disk payloads after terminal receipt.
Sim/POC/Prod E2E tests must exercise these boundaries, including process
termination and filesystem failure after completion publication.

Released attempts leave the supervisor cleanup set and drop their owned handles.
Lightweight launch identities remain for the job lifetime to prevent relaunch.
