# Task execution mode: architecture and lifecycle

Task execution mode releases compute resources between federated assignments while
keeping federation coordination alive. A job uses `execution_lifetime: "job"`
(the default) for resident compute, or `"task"` for a fresh compute worker per
assignment. This document defines the roles, resource contract, execution flow,
storage, failure handling and current limits. The Kubernetes operator guide gives
deployment commands and troubleshooting; it is not required to understand this design.

## Roles and assumptions

| Role | Job lifetime | Task lifetime |
| --- | --- | --- |
| Client Parent (CP) | Starts and monitors client jobs. | Starts and monitors client jobs; owns Kubernetes task Pod creation and cleanup. |
| Client Job (CJ) | Keeps coordination and compute resident. | Keeps federation communication, filters, analytics publication and result submission resident. |
| Task worker | No separate task runtime. | Constructs compute, executes one assignment, finalizes and exits. |
| Server Job (SJ) | Issues assignments and receives results. | Uses the same assignment and result-receipt protocol. |

The current site profile uses one backend throughout CP, CJ and workers: Process
uses processes; Kubernetes uses Pods. CPU or GPU is an independent resource choice
within a capable backend. Site configuration selects the backend and its authority;
a job cannot supply runtime credentials, accounts, namespace or arbitrary mounts.
Export validates resource shape; the receiving backend must enforce the requested
resources or reject unsupported requirements before compute.

Each TaskSupervisor admits one active physical attempt. The Kubernetes CP also
admits at most one unsettled attempt per job. Worker memory, optimizer state and
executor instances do not survive an assignment. Application-selected checkpoint
files can survive. Process CJ recovery can replace a failed CJ only after its job
handle confirms termination. This Kubernetes profile does not opt into CJ restart:
its CP still fences unsettled attempts and refuses retained subtree adoption.
Persistent files alone do not provide CP restart adoption, physical task retry or
distributed training.

Four separate facts govern the lifecycle:

| Fact | Meaning |
| --- | --- |
| Local completion | An immutable record binds the assignment identity to validated output. |
| Physical settlement | The worker and its owned execution have stopped; output writers are quiescent. |
| Server receipt | The server has received the complete result, or has closed the assignment. |
| Aggregation acceptance | The algorithm decides whether and how to use the result. |

None of these facts implies the others. In particular, sending a result does not
settle a worker, and receiving a result does not promise aggregation or recovery
after server restart.

## One assignment from delivery to cleanup

1. SJ issues an assignment with workflow, task and attempt IDs. Retransmission
   preserves those IDs; a new physical attempt requires a new server-issued attempt ID.
2. CJ applies input hooks/filters and stages input plus an inert worker bootstrap.
   Compute definitions are authorized in CJ and again in the worker before custom
   imports, component construction or input decoding.
3. The trusted launcher acquires a worker with the compute resource budget. CJ
   retains federation communication and can receive cancellation while it waits.
4. The worker initializes compute, executes once, finalizes, seals output and any
   checkpoint candidate, and publishes an immutable completion record.
5. The supervisor confirms physical settlement and validates completion identity,
   worker PID, payload sizes and digests before consuming output.
6. CJ reports diagnostics and replays worker logs. Metrics have already been
   delivered live through the worker's Cell session. Only after those
   fallible steps succeed does it promote a valid optional checkpoint locally,
   followed by downstream result processing/filters and submission.
7. CJ retries submission of that saved result until an identity-bound terminal
   receipt or closure arrives. It does not rerun compute to recreate a lost reply.
8. Successful attempt payloads are released only after settlement, terminal receipt
   and reader quiescence. Failed or unresolved attempts retain their evidence.

Cancellation takes precedence over network publication. A local completion never
authorizes sending after CJ/task cancellation. Shutdown closes admission, drains
pending launches and cancels late-acquired handles; losing a launch reply does not
make the acquired execution unowned.

## Coordinator and compute resource contract

| Job metadata | Job lifetime | Task lifetime |
| --- | --- | --- |
| `resource_spec` | Resident CJ resources, including compute. | Resources for each compute worker. |
| `cj_resource_spec` | Rejected for resident client apps. | Independent resident coordinator budget. |
| `launcher_spec` | Backend-specific runtime settings. | Backend-specific runtime/compute settings; compute resource fields do not become CJ resources. |

Resource admission uses the configuration-only `get_client_execution_lifetime`
helper at export, submission, CP/CJ startup and Kubernetes profile selection.
It currently validates the top-level `execution_lifetime` as `job` or `task`,
defaulting to `job`. Executors with other lifetime declarations extend this one
classifier; this does not change `prepare_task_execution`'s executor-wrapping
rules or enable additional executors by itself.

A GPU task job can use a CPU coordinator. For example, its job metadata can use:

```json
{
  "name": "gpu-task-job",
  "min_clients": 1,
  "deploy_map": {"app": ["@ALL"]},
  "resource_spec": {
    "@default": {"num_of_cpus": 2, "memory": "4Gi", "num_of_gpus": 1},
    "server": {"num_of_gpus": 0}
  },
  "cj_resource_spec": {
    "@default": {"num_of_cpus": 0.25, "memory": "512Mi", "num_of_gpus": 0}
  },
  "launcher_spec": {
    "default": {"k8s": {"image": "registry.example/nvflare-training:2.10"}}
  }
}
```

The client app declares `execution_lifetime: "task"`. The server keeps its existing
resource semantics; its explicit zero-GPU override above prevents the compute GPU
default from also applying to SJ. In Kubernetes, the job's image setting selects
the CJ image. The worker image is fixed separately by trusted site task policy.
Job-supplied images retain existing bring-your-own-code authorization requirements.

Portable compute CPU counts are positive integers; CJ CPU shares may be positive
fractions. Native Kubernetes CPU quantities can also express fractions. Portable
memory uses positive integer Mi/Gi/Ti quantities. GPU counts are nonnegative
integers, and explicit zero overrides a GPU site default. A GPU worker also needs
CPU and memory.

Resource resolution has explicit boundaries:

- `resource_spec` and `cj_resource_spec` use `@default`; site overrides are merged
  per field. Coordinator resolution also overlays a matching `@ALL` entry before
  a named site. `launcher_spec` instead shallow-merges `default[mode]` with the
  named site's mode block. `default` and `@default` are distinct keys.
- Export/submission derives an empty coordinator entry for each task site when
  needed. That entry selects site defaults and removes compute GPU requirements
  from whole-job admission. User metadata cannot mark a resident app as task based.
  CP and CJ check the locally resolved lifetime before launch/component construction.
- Missing coordinator fields never inherit compute requirements. Kubernetes CJ
  defaults are 0.25 CPU, 512Mi and zero GPUs, configurable through site
  `ClientK8sJobLauncher.cj_resources`. Size CJ memory for filters and result handling;
  these are resource defaults, not a workload sizing guarantee. Explicit CJ GPUs
  are independent of worker GPUs.
- Portable and native declarations of the same resource conflict. Legacy nested
  resource settings remain a compatibility fallback when the matching launcher
  block is absent; legacy/native GPU counts must remain consistent.
- Submission preserves `@default`/`@ALL` through CLI preparation and serialization.
  Server-side lifetime inspection does not resolve unrelated client environment
  variables or open client HOCON includes. Declare the root lifetime literally.

Kubernetes enforces each worker's Pod requests/limits. Two CPU CJs at the same site
can stay resident while their workers successively use one available GPU.
Kubernetes chooses scheduling and fairness; there is no round-robin guarantee.
Unsupported licenses, custom resource-manager claims and per-GPU memory quotas
must be enforced or rejected, never silently discarded. The current Process path
rejects unenforceable compute CPU/memory and custom claims before launching a
normalized task CJ and again before execution. It has no implicit CPU/memory
limiter, Kubernetes coordinator defaults or GPU allocation ledger. An explicit
site-owned Process gpu_devices list exposes the first N configured devices to a
worker requesting N GPUs, while its task CJ remains GPU hidden. This supplies
static visibility; it does not exclude other jobs or host processes. See the
[Process GPU profile](process_task_gpu_profile.md) for configuration, supported
resource declarations and failure recovery, including the fence after abrupt
CJ death when disposable worker settlement cannot be inferred.

## Application and site configuration

Task client exports use `format_version: 3`, so older clients reject them instead
of silently running resident compute. Job lifetime keeps version 2. Authors use
`FedJob(execution_lifetime="task")`, `job.set_execution_lifetime(...)`, or supported
recipe/client configuration. Named apps can have different lifetimes at different
sites; an `@ALL` app cannot also have a site-specific lifetime override.

The supported compute profiles are in-process ClientAPIExecutor, StatisticsExecutor
and NPTrainer. Executors requiring resident federation engine services use job
lifetime. Statistics workers construct only the generator named by `generator_id`,
which must initialize from local data on every assignment. Filters, widgets,
receivers and privacy processing remain in CJ. Custom decomposers are registered
in both runtimes before decoding; that does not move compute into CJ.

Workers construct authorized inert executor/component specifications directly,
using `initialize`, `execute` and `finalize`. Finalization also runs after partial
initialization or failure, although forced termination cannot guarantee it.
Workers do not dispatch application lifecycle events or build nested component
graphs. `component_dependencies` and component `execution_scope` are rejected;
the existing `args.class_attributes` wrapper and ordinary `config_type: "dict"`
constructor data remain supported. Authorization precedes application logging,
custom imports, decomposer setup and input decoding.

A Client API task script runs on the worker's main thread, receives one input and
sends one result through the shared InProcessClientAPI, then exits. Script arguments,
FULL/DIFF exchange, metadata and rank semantics are preserved. Sending durably
stages output; script/finalizer failure before completion still fails the attempt.
Metrics travel live over a worker-to-CJ Cell session. CJ fires each metric into
its active attempt's context before replying; unsupported/rejected metric data
fails the worker, while tracking-receiver exceptions are logged as in job mode.
A metric request waits up to 30 seconds for CJ's reply. The session closes after
physical settlement and rejects later metrics. Analytics are not stored in an
attempt spool or replayed on resubmission. Worker logs are replayed through CJ
logging after settlement, bounded to 1 MiB and 4,096 lines;
the complete file remains available for diagnosis. The first failure retains its
type, message and diagnostic path; a later cleanup failure cannot replace it.

SCAFFOLD clients do not restore their control variates between workers and are
rejected. Local-model submission, external/attach Client API sessions and the
resident Collab RPC service use job lifetime. Global-only cross-site evaluation
is supported. Lightning's in-memory `restore_state=True` cache cannot restore a
fresh task worker; use explicit native checkpoints or job lifetime.

Only canonical site `local/resources.json` (or its default fallback) supplies
`task_execution`. Jobs cannot select the internal adapter, launcher, environment
policy or task state. A Process site can configure:

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

Process workers receive the framework allow-list plus approved environment names;
federation credentials, API bootstrap and arbitrary GPU visibility are excluded.
Native worker environments come from trusted backend configuration, rather than
Process-style forwarding. CPU workers mask GPU visibility; GPU-capable backends
arrange devices through their resource binding. `artifact_cleanup: "retain"`
keeps attempts. Optional positive `max_payload_bytes` applies consistently to
input, bootstrap policy and output, including all referenced tensor files.

## Backend ownership and physical settlement

Both backends use `task_worker_process`. Process scope requires an explicit live
parent PID greater than one and an owned process group; native scope does not
mistake CJ for the worker's OS parent and supports PID 1. Process supervision
requires Linux or macOS with Python 3.13+ waitid/WNOWAIT. CJ owns Process group
cleanup; this profile does not promise independent CP cleanup after abrupt CJ death.

In Kubernetes, the CJ's K8sTaskLauncher is a tokenless CellNet proxy. Before CJ
starts, CP installs a backend bound to canonical site policy and admitted job/site
resources. Bounded JSON launch/poll/cancel commands use a per-job capability token;
the sender-set origin header is an additional consistency check. CJ cannot choose
Pod manifests, Kubernetes credentials, namespace, account, mounts or another job's
attempts. The CP backend creates K8sTaskHandle; CJ sees a remote handle. Job and
task launchers share stateless Pod helpers while keeping distinct lifecycle handles.

CP owns accepted attempts even if their create or reply is still in flight. Normal
job termination closes task admission and deletes CJ, including a never-started
Pending CJ. CP keeps task cleanup after CJ exits. CJ end-run makes bounded cleanup
calls and can exit while reconciliation continues. A ten-second independent CJ
observer backs up normal job-handle polling.

Deleting a Pod does not prove execution stopped. Task deletes use UID preconditions;
replacements are not adopted or deleted. A finalizer keeps the owned Pod observable
until a scheduled Pod has terminal status, all admitted containers have terminated
and kubelet sandbox teardown is confirmed, or an unscheduled Pod's deletion fences
scheduling. Only shutdown evidence plus confirmed Pod removal establishes settlement.
Transient API errors preserve the last-known phase. Missing/replaced Pods, node loss
or ambiguous create replies without evidence leave ownership fenced; a timeout or
404 alone cannot authorize retry or result publication.

Unknown cancellation identities are retained to fence delayed launches, capped at
128 per job. Trusted `max_attempts` (default 10,000) bounds all retained identities,
including cheap rejected launches. Exceeding either limit closes admission and
cancels known execution without evicting replay fences. Existing poll/cancel and
CP cleanup remain available. Released supervisor attempts drop heavy handles;
lightweight launch identities remain until job shutdown.

## Payloads, checkpoints and receipts

Workers and CJ exchange small FOBS serialization records and checksum-bound files
on a shared job filesystem. PyTorch tensors use adjacent per-tensor safetensors
files. Unfiltered CJ input hooks receive local lazy references with metadata and
byte sizes, rather than tensors supporting arithmetic; workers receive mapped
tensors. If input filters apply, CJ resolves inputs eagerly before task-data events.
CJ result callbacks/filters receive mapped tensors, and their edits are transmitted.
Clearing `TASK_DATA` before result filters keeps its existing ordering.

Single-tensor receive files are hard-linked when possible; otherwise serialization
writes one tensor at a time. Receive scratch files are released in `finally`.
Temporary serialization scales with the largest tensor and sender cache, not the
entire model; mapped pages and filesystem cache still grow with model size, writes
can allocate private pages, and per-tensor files consume inodes. There is no fixed
4 GiB ceiling. `max_payload_bytes`, when configured, counts the serialization record
and all tensor files together; other metadata limits remain separate.

The store uses exclusive publication, real hard links, file/directory fsync and
atomic small-manifest replacement on one filesystem. Hashes are computed while
writing and verified before completion publication and before decoding. Bootstrap
files and attempt artifacts cannot be symlinks; CP confines mapped worker paths to
the configured job subtree. Interrupted publication temporaries can be removed
only after writers/readers are quiescent. The storage preflight is not proof of a
driver's cross-Pod coherence or crash durability; deployment qualification checks both.

Completion publication is serialized with fatal, abort and direct-failure
reporting. An observed failure or direct compute/analytics operation still in
flight prevents publication. Successful publication seals the worker runtime
against later operations and abort requests; returning from `flare.send` alone
does not establish completion.

### Application checkpoint state

Scripts choose which model, optimizer, data-loader and RNG state to persist.
`flare.system_info()["task_checkpoint"]` supplies an isolated, verified `input`
copy of the last committed checkpoint (or None) and an attempt-unique pending
`output` path. The script uses its own framework to load/save that opaque file.
The worker seals it after finalization and binds its checksum to the completion.

After settlement, output validation and diagnostic/log replay, CJ atomically
promotes the candidate with the completion identity and result reference, before
result submission. Missing, invalid or cancelled candidates, and failures before
checkpoint manifest replacement (including replay failure), leave previous state
unchanged. Promotion is idempotent. Manifest replacement is the local commit point:
a later directory-sync/cleanup error retains that checkpoint and its matching
result, with a warning that crash durability was not confirmed. A later nonzero
worker exit or diagnostic likewise does not erase an already verified completion;
execution status and evidence remain separate, and failed attempts retain payloads.

The committed payload is linked from the settled attempt; superseded links are
removed and one checkpoint remains per job/executor/task slot. A later worker gets
one isolated input copy, so changing that copy cannot corrupt committed state.
Resubmission reuses saved output without re-running training, replaying analytics
or promoting again. Only attempts that finished CJ result processing are eligible;
missing, corrupt, unsettled or released saved results fail rather than re-execute.
Switching an arbitrary stateful script to task lifetime does not make it equivalent
to a resident script; the application must explicitly preserve its required state.

### Result receipt and payload release

CJ validates workflow/task/attempt identity on every result reply:

| Reply | Action |
| --- | --- |
| `RECEIVED` | Stop resending; the server owns the complete payload. |
| `TASK_CLOSED` | Stop resending; the assignment is no longer wanted. This does not prove an earlier upload was unused. |
| `RETRY`, lost, malformed or mismatched reply | Retain the saved result and retry submission, without compute. |

The server validates the authenticated site/job and assignment identity before
filters, callbacks and aggregation. Duplicate submissions do not repeat those
effects. Receipts also cover failed or filtered-out results; they do not promise
aggregation acceptance or durable receipt history across server restart. Local
checkpoint promotion therefore does not wait for algorithm acceptance or receipt.

Default payload cleanup requires settlement, decoded successful output and a
terminal receipt/closure. Saved-result readers pin files while verifying/decoding
or mapping them, so concurrent receipt cleanup waits. Existing POSIX mappings
survive unlink; additional readers that reopen filenames must arrange ownership
before cleanup. There is no general reader-lease API. Failure, explicit retention,
unknown receipt or unresolved settlement keeps evidence. Completion/diagnostic
records and the separately linked committed checkpoint survive successful release.

## Kubernetes namespaces, storage and isolation

All CJs and their task Pods use the launcher's configured jobs namespace. CP can
use that namespace or a separate control namespace with explicit permissions and
a compatible storage arrangement. Workers always use their CJ's namespace; the
launcher does not create a namespace per job or choose one from job metadata.

The job launcher accepts an optional `startup_secret_name` shared by its CJ and
worker Pods. The default is `nvflare-job-startup-<normalized-participant>`.
This public-only Secret is separate from the deployment-staged CP Secret
(`nvflare-startup-<normalized-participant>`), whose encoded keys and private
site credentials must stay unchanged. A `startup_secret_name` override must name
a dedicated public job kit, not CP's staged kit. Existing launcher-created
Secrets with canonical filename keys can be retained using that override.
Operators sharing a namespace across federations must configure unique Secret
names per federation and participant. Each startup Secret carries the root CA's
SHA-256 fingerprint as a label-safe base32 identity. Replacement checks that
identity and the observed resource version; another federation's Secret causes
launch to fail. Identical kits are reused without replacement, and concurrent
updates re-read and revalidate ownership with a bounded retry count. An unlabeled legacy Secret can be upgraded only after matching
its CA. Unsecured kits without a root CA use a digest of the public kit payload.
Changing a trust root or unsecured kit requires a fresh Secret name or deliberate
operator-managed replacement after stopping its consumers.

Workers connect directly to CJ's live Cell listener, on a port selected by CJ.
CP injects `NVFLARE_K8S_CJ_POD_IP` from `status.podIP`; K8sTaskLauncher advertises
that IPv4 address rather than loopback or the CP's configured address. Cluster
routing and NetworkPolicy must permit worker-to-CJ traffic on that selected port.
CellNet currently requires IPv4 for this path. The listener keeps the site's
internal scheme/security. In mTLS mode the bootstrap supplies the job identity,
CA and both client/server certificate-key pairs; credential paths are identical
inside the CJ/worker shared workspace. A startup kit without the site's private
key remains valid because both Cell roles use the per-job credential. See the
optional [Cell session details](task_worker_cellnet_session.md).

The usual site layout has a private CP workspace PVC plus a separate shared jobs
PVC. CP mounts the jobs claim and creates a fresh hashed job/site directory; each
CJ/worker mounts only that directory as its workspace subPath. Task execution
requires this durable profile. Resident jobs keep emptyDir unless the site enables
`persist_resident_workspaces: true`, which stores their checkpoints on the same
jobs claim. No claim is provisioned per job/task. PVCs are namespaced; control/jobs
namespace separation needs an operator-supplied storage arrangement.

CP, CJ and workers share configured nonroot UID/GID. Mode 0700 does not separate
same-owner jobs: cross-job filesystem isolation relies on kubelet subPath mounts
and the storage driver. Jobs never mount CP's private claim. Trusted templates
cannot add a second full jobs-claim mount. Site seccomp/supplemental-group settings
are preserved while required nonroot/no-escalation settings are enforced. CJ and
workers receive no Kubernetes API token. This profile does not allocate per-job
UIDs/claims or isolate worker access from files already inside its own job subtree.
Additional CJ dataset volumes are not automatically inherited by task workers.

Persistent job subtrees are retained after success as well as failure. Operators
budget capacity, archive and remove them only after the job, runtimes, in-flight
creates and readers/uploads are quiescent. Reusing an existing job/site subtree
fails instead of adopting or overwriting it. Definitive pre-create/API rejection
removes only a freshly reserved, unchanged empty directory; ambiguous outcomes keep
their storage fence. There is no automatic subtree collector or Kubernetes CJ/CP recovery.

## Deadlines and failure handling

Elapsed deadlines use a monotonic clock; wall-clock timestamps are diagnostic.

| Setting | Scope |
| --- | --- |
| `queue_timeout` | Time while a Kubernetes Pod is unassigned to a node; default None permits indefinite queue waiting. Diagnosed scheduling/admission failures can fail earlier. |
| `startup_timeout` | Time from first observed Kubernetes node assignment until Running; default 120 seconds. |
| `worker_timeout` | Optional absolute whole-worker lifetime, including queueing and artifact I/O; a job may shorten it. |
| `result_wait_timeout` | Client API application execution through script exit/finalizers; framework result serialization pauses this deadline. |
| `finalization_timeout` | Finite post-staging application finalization bound, default 30 seconds. Later checkpoint sealing and publication remain subject to the absolute worker deadline. |
| `settlement_timeout` | Supervisor's bounded wait for terminal execution to settle, default 10 seconds; expiry fences further attempts and retains ownership/evidence. |
| `cleanup_timeout` | Kubernetes cleanup-call bound; CP can continue reconciliation after CJ exits. |

Resident job-mode Kubernetes `pending_timeout` remains a separate existing policy.
Exceeding a deadline does not prove settlement. The supervisor has no reset or
automatic physical retry after unconfirmed settlement. CP crash/restart, permanent
node loss or missing finalizer evidence requires operator reconciliation after
verifying runtime shutdown; clearing a stuck finalizer is not a substitute for proof.

## Qualification and optional references

Qualification covers CPU/GPU and job/task modes, actual submitted resource
requests, two CJs at one site sharing one available GPU, cancellation while queued
or running, never-started CJ abort, CJ loss, lost replies and API outages. It also
checks immutable output after settlement, checkpoint continuity, receipt retry
without compute, active-reader retention, mounts/UIDs and namespace boundaries.
Record source/image identity and distinguish unit evidence from cluster results.

The following references provide deeper protocol/API detail; the contracts needed
to understand task execution are stated above:

- [Kubernetes operator guide](../user_guide/admin_guide/deployment/kubernetes_task_launcher.rst): deployment, sizing, retention and troubleshooting.
- [Job metadata reference](job_meta_launcher_config.md): shared schema and backend-specific settings.
- [Assignment/receipt protocol](task_assignment_admission.md): wire compatibility, hierarchy and server history.
- [JobLauncher and JobHandle reference](job_launcher_and_job_handle.md): resident runtime APIs and other launchers.
