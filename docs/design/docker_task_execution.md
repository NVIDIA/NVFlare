# Docker task execution: CPU and GPU contract

The shared [task execution design](task_worker_lifecycle.md) describes assignment
handling, result receipts, payload retention and worker settlement. This document
covers Docker daemon authority, bind mounts and GPU admission.

For the setup sequence, image checks, and reuse of an existing host after a
source fix, see [Docker task worker setup and test procedure](../user_guide/admin_guide/deployment/docker_task_execution.rst).

Original v5 deployment qualification base: upstream `1b5e7f9832e6a777c54e22c72e6631bbd344babe`, including
P4 merge `94c5b0aebd3de9a6733926d5fc285144bfd335da`.

* **Authority:** Docker CP retains the daemon connection already used by
  `ClientDockerJobLauncher`. CJ uses a job-scoped, authenticated Cell proxy.
  Requests identify staged attempts; they cannot supply Docker commands, images,
  daemon endpoints, mounts, credentials or device requests. Site configuration
  selects the worker image, Python executable and environment. No Docker socket
  or backend credentials enter CJ or worker containers.
* **Identity:** one container per authority-issued job/site/task/attempt identity,
  fenced for the lifetime of its CP session. A unique name and ownership label
  bind the execution to that session. Transport retries recover the existing
  handle; they never launch compute again. A lost creation acknowledgement is
  unresolved ownership, never permission to retry physical creation.
  CP retains at most 10000 attempt identities per job by default, configured by
  `task_worker_config.max_retained_attempts`. Rejected and cancel-before-launch
  attempts count too. The CJ proxy has its own `max_retained_attempts` argument
  with the same default. Exhaustion closes admission without evicting replay
  fences; existing handles remain available for cleanup. Repaired bootstrap files
  do not authorize another launch of a rejected identity.
* **Storage:** single-daemon Linux bind mounts. The persistent daemon-host job
  directory appears at `/var/tmp/nvflare/workspace/<job_id>` in both CJ and worker.
  Bootstrap paths and opaque checkpoint records retain their existing format.
  Startup and site configuration are read-only mounts; only this job directory
  is writable. Dataset bindings come from site-owned study configuration. The
  same numeric UID/GID as CP preserves mode-0600 artifacts. Different worker
  workspace paths, remote artifact transfer and ephemeral-only storage are
  outside this first profile. Host source paths are never inferred from CJ paths.
  The accepted site layout separates private CP storage from shared job state;
  Docker's current single workspace-root configuration still needs that alignment.
* **Resources:** CPU or GPU workers, with independent CJ and compute budgets.
  CJ CPU/memory budgets replace inherited resident quota/period, memory
  reservation and swap-limit controls, which can conflict with Docker's native
  validation or defeat the requested CPU cap. Resident jobs retain those controls.
  Each job session admits at most one unsettled execution. These limits do not
  reserve an exclusive share of the host CPU pool. Workers join CJ on the
  CP-selected user-defined Docker network; this profile uses local mounted data.
  CJ defaults to CPU coordination; an explicit CJ GPU budget has job-lifetime
  ownership. CPU workers have no GPU access. GPU workers receive physical UUIDs from
  a site-approved exclusive partition. CP admits devices using daemon-wide
  ownership records before creating the worker; device visibility is not admission.
* **Settlement:** terminal container observation, owned removal and confirmed
  absence from the same daemon establish execution-unit settlement. GPU
  settlement additionally requires confirmed release of every owned device
  reservation, after worker removal. Stop
  responses, PID-1 exit, transient API failures and unexpected disappearance do
  not establish settlement. Docker must enforce its ordinary private PID/cgroup
  namespaces; privileged execution, host PID namespaces, restart policies and
  container escape are unsupported. Capture bounded daemon logs before removal.
* **Publication:** CJ retains filters, tracking, checkpoint promotion, receipts
  and publication. It reads validated committed output only after settlement.
  Container removal never deletes bind-mounted task artifacts. Existing
  supervisor receipt/reader/retention rules remain authoritative; failed or
  unresolved attempts retain diagnostics. No remote exactly-once guarantee,
  automatic physical retry or CP restart adoption is introduced.

## CP authority and cleanup

The task control channel accepts bounded JSON objects, a per-job capability,
and the exact CJ Cell origin. Its four operations address an attempt within
that job; there is no API for caller-selected container IDs, images, commands,
mounts, environment or Docker endpoints. CP reads the staged JSON bootstrap
with `read_bootstrap(..., resolve_parent=False)`. Directory-descriptor traversal
rejects symlinks in every job-controlled parent and the bootstrap file itself;
CP does not canonicalize those paths into another job's directory.

Task CJs reject Docker control environment settings, host/other-container
namespace sharing, extra capabilities and device/socket inheritance. They use
the fixed `cap_drop=ALL` and `no-new-privileges` policy, including overrides of
Docker endpoint settings baked into images. Site administrators must keep
approved dataset and study-secret bindings free of CP runtime credentials.

CP registers a retained owner before CJ creation and an execution record before
each worker/reservation create. Native API calls run outside the session's
admission lock, so cancellation can close admission while a create/start reply
is delayed. An in-flight call remains unsettled; its eventual return observes
the cancellation request and cleans up the owned native objects.

A CP monitor owns each task job's CJ and worker cleanup independently of CJ
RPCs. CJ containers use `auto_remove=False` so CP can observe their identity and
exit, remove only an owned container, and confirm absence. On CJ exit or abort,
the monitor closes admission and retries task/GPU cleanup after API recovery.
Job `poll()` stays UNKNOWN and `wait()` retains job files and credentials until
both CJ and task cleanup are confirmed. Returning from `terminate()` only
acknowledges a cleanup request. CJ exit with unfinished tasks cannot become a
successful job outcome merely because cleanup later succeeds.

Both resident and task Docker launchers publish the created job handle through
the shared `publish_job_handle(handle, fl_ctx)` helper before waiting for
RUNNING. The client executor can then apply admin or heartbeat abort while CJ
is still starting, including an abort recorded before handle publication.
Launchers without a publication callback still return their handle normally.
This does not interrupt an outstanding Docker create/start API call; CP's
retained owner continues to reconcile its outcome.

The live [worker Cell session](task_worker_cellnet_session.md) uses the CJ
container's Docker DNS name, supplied by CP through
`NVFLARE_DOCKER_TASK_LISTENER_HOST`. Workers and CJ share the launcher's named,
user-defined network; `host`, `none`, Docker's default `bridge`, and
`container:...` network modes are rejected for task workers. The CJ listener
inherits the site's internal scheme and connection security with a fresh port.
A worker must connect and complete HELLO before computation. Metrics use this
live session; there is no analytics spool or post-exit metric replay. In secure
mode, bootstrap pins both client and server TLS roles to the job certificate
and key, plus the mounted public CA; it never selects site private keys.

This ownership allows bounded CJ END_RUN handling without an infinite CJ
retention loop. CP process death/restart adoption remains outside this profile;
uncertain native objects and reservation records require operator reconciliation.

## Persistent storage and resume

The accepted default site layout has two stores, shared across that site's jobs:

* A private CP workspace holds site credentials, runtime configuration and
  parent state. CJ and worker containers must not receive this workspace mount
  or broad runtime credentials.
* A separate jobs-state store holds per-job directories and is also mounted by
  CP. Each CJ and its workers receive only that job's directory, alongside the
  approved read-only startup/configuration files and dataset bindings. A new
  store is not required for every job or attempt.

Docker currently derives both its private files and per-job bind sources from
the launcher's single daemon-host `workspace` root. CJ and workers receive the
individual job bind, not that whole root, but there is no separate jobs-state
root setting yet. The two-store default remains deployment/launcher alignment
work before readiness; these docs do not introduce a new configuration option.
Do not mount the full CP workspace into a job to emulate shared storage.

For both resident (`execution_lifetime="job"`) and task lifetimes, applications
must persist required checkpoints and recovery metadata inside the job bind.
Task bootstrap, immutable outputs, receipts and committed checkpoints remain
subject to their existing artifact retention rules. Container removal preserves
these bind-mounted files; it does not preserve the container's writable layer
or the workspace-root tmpfs. In particular, root-level `snapshot-storage` or
`jobs-storage` directories created outside the job bind are ephemeral.

Resident Docker jobs use main's [automatic CJ recovery](client_job_recovery.md)
when confirmed container exit and the shared recovery policy permit it. CP
refreshes launch tokens, retains the logical job's workspace, certificate and
resource reservation, and publishes each replacement handle before startup
waits. `max_cj_restart_failures` in `local/application.conf` defaults to three
consecutive failed replacement initializations; READY resets that count. CP and
the server must remain running, and applications restore their own checkpoints.

Task-mode Docker CJs return `can_restart=False`. Server redelivery keeps the
same task/attempt ID, while the interrupted attempt's directory and CP replay
fences remain retained. A new CJ cannot resume that assignment merely because
its predecessor's containers were removed. Even an empty CP execution map is
insufficient: CJ may have staged files before sending its launch RPC. Supporting
this requires a shared durable task recovery protocol; deleting artifacts,
evicting fences or inventing a fresh physical attempt is unsafe. CP still
settles owned workers and GPU reservations before reporting the task job's
failure. CP restart adoption remains unsupported for both lifetimes. Preserve
required job state and unresolved-settlement evidence until reconciliation and
the existing reader/retention rules permit cleanup.

## Shared integration boundary

Merged Kubernetes PR #5490 supplies the shared contracts below. Docker uses
those main-branch implementations and adds its CP authority and resource resolver:

1. `nvflare/private/fed/client/task_execution_policy.py`: select explicit native
   Docker, Kubernetes and Slurm classes using lazy imports. Each backend's
   `validate_site_config(site_config)` checks the canonical site's matching CP;
   retain Process as the default and reject unsupported/mixed profiles.
2. `nvflare/private/fed/client/task_worker_executor.py`: deliver its existing START_RUN context to the
   selected launcher's existing FLComponent event hook so a proxy can bind the
   CJ Cell; pass the site's resolved `TaskResourceRequest` to the existing launcher.
3. `nvflare/job_config/task_execution.py` and `fed_job_config.py`: export validates
   GPU counts without selecting the receiving site's backend. Site runtime
   decides capability/admission. `validate_task_resources(..., allow_gpu=False)`
   preserves the existing Process GPU guard; supported native backends enable GPU
   requests and admit them through their authority. `TaskExecutionPolicy.task_resources` is the
   narrow shared resolution hook for the separate Process GPU implementation.

Native workers receive no Process environment passthrough. Their immutable
START_RUN binding owns the native resources. Docker carries CP-bound compute
CPU/memory/GPU requirements in the existing TaskResourceRequest. Shared export
and server validation derive task sites from actual client configs, including
mixed lifetimes and `@ALL`, and normalize `cj_resource_spec`. The long-lived
resource manager admission then covers CJ only. Local CP and CJ startup bind
that admission to the resolved execution lifetime before app components load.

A separate shared fix pins saved-result payloads while readers decode them.
Terminal receipts defer cleanup until the last reader finishes; failed reads
retain evidence. Compute admission remains independent of result readers.
TaskLauncher/TaskHandle and supervisor/artifact/checkpoint interfaces stay unchanged.

Docker uses a separate container entry point, preserving the Process entry
point's parent-PID/process-group checks. Kubernetes/Slurm need their own trusted
authority/storage design; accepting a backend name alone must not bypass guards.

## Qualification status

CPU and GPU are both implementation scope. On 2026-10-21, the frozen v5 serial
fixture completed all four CPU/CUDA x job/task rows on two clients with two
rounds each. CPU rows reported zero visible GPUs; CUDA rows reported one GPU
per client, a checksum of 33554432, and peak allocation of 67111936 bytes.
The source and both images matched the recorded runtime file manifest.

The earlier v4 CUDA resident control failed because its count-based request
conflicted with the site's exclusive task GPU partition. That failure remains
recorded under its original identity; v5 used separate resident and task site
policies and new job submissions. The v5 results apply to that frozen candidate;
later shared-runtime changes require deployment qualification again.
Cancellation, uncertain settlement, GPU contention/reuse, retained artifacts,
and secure deployment qualification remain separate acceptance requirements.

## Job credential boundary

The worker receives no Docker socket or provisioned site private key. However,
the inherited writable job bind includes the job directory, and a secure job's
`job_cert/job.key` is stored there for CJ. With the same numeric UID, the worker
can access that job-scoped credential. The live internal mTLS session
intentionally uses this job credential. Clearing credential environment variables
does not isolate the file. This profile therefore
does **not** claim a credential-free worker filesystem or isolation from a
malicious same-job worker. Changing that boundary requires a separately reviewed
mount/storage design; it is not part of a qualification setup correction.

## Site configuration

Prepare the normal Docker runtime kit using `nvflare deploy prepare`. Then add
the following fields to the client's canonical `local/resources.json` (or its
`resources.json.default` fallback). Keep its other existing components. Repeat
this for each client site; server jobs continue to use `ServerDockerJobLauncher`.
Deploy preparation currently accepts only the resident Docker settings, so add
these task settings **after** preparing the kit.

```json
{
  "components": [
    {
      "id": "docker_launcher",
      "path": "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher",
      "args": {
        "network": "nvflare-network",
        "task_worker_config": {
          "image": "site-approved-worker:reviewed-source",
          "python_path": "/usr/local/bin/python",
          "environment": {"OMP_NUM_THREADS": "1"},
          "nano_cpus": 1000000000,
          "mem_limit": 1073741824
        }
      }
    }
  ],
  "task_execution": {
    "launcher": {
      "path": "nvflare.app_opt.task_launcher.docker_launcher.DockerTaskLauncher"
    },
    "worker_timeout": 120,
    "settlement_timeout": 60,
    "artifact_cleanup": "retain"
  }
}
```

The image must contain this implementation and the application's dependencies.
Install the Docker Python SDK in CP's image; CJ and worker images do not need it.
An image-selected entrypoint is cleared so the fixed Python worker becomes PID 1.
Use a non-root CP user with daemon access and matching numeric IDs in all three
containers. Workers get the existing public startup-file mounts and site policy,
not the provisioned site private key. Do not add a socket bind, `volumes_from`,
privileged mode or device mounts to CJ. Study dataset binds are reused; study
secret mounts and CJ environment are not implicitly forwarded to workers.
Set worker environment explicitly in `task_worker_config.environment`;
`task_execution.environment_variables` applies to Process and is rejected here.
The worker has a private `/tmp`; HOME is `/tmp` for UID portability.

Jobs keep the single application flag `execution_lifetime="task"`, and ordinary
jobs retain the default `"job"`. Existing job metadata selects the CJ image as
before. Worker image and command policy remain site-owned. Existing
`resource_spec` and native launcher compute fields describe CJ in job lifetime
and workers in task lifetime. Portable CPU/memory map to worker `nano_cpus` and
`mem_limit`; GPU counts can be portable or Docker-native, subject to the shared
conflict checks. Legacy compute GPU counts use `get_compute_resource_manager_spec`,
not the CJ admission resolver. Old task metadata that would still reserve
compute GPUs for the whole job must be re-exported/resubmitted.

`cj_resource_spec` supplies an independent per-site/`@default` CJ budget. For
example, task metadata can request:

```json
{
  "resource_spec": {"@default": {"num_of_cpus": 2, "memory": "4Gi", "num_of_gpus": 1}},
  "cj_resource_spec": {"@default": {"num_of_cpus": 0.5, "memory": "1Gi", "num_of_gpus": 0}}
}
```

Shared export/submission adds the actual task-site entries. Server requirements
must override GPU count to zero when the server uses no GPU. Do not also set a
native GPU count for the same portable requirement. Omitted CJ fields use the
trusted launcher's `cj_resources` defaults: 0.25 CPU, 512Mi, zero GPUs. Fractional
CJ CPU shares map to Docker nanocpus. This budget is independent of resident
job defaults and worker defaults. Omitted worker CPU/memory use
`task_worker_config.nano_cpus` / `mem_limit` (defaults: one CPU and 1Gi), including
GPU workers. CPU limits are cgroup quotas, not exclusive host CPU reservations.

CP snapshots each job's compute budget, verifies CJ requests against it, and
enforces the limits on every physical worker. Custom RM claims and positive
per-GPU memory claims are rejected because this profile cannot enforce them.
An explicit CJ GPU request uses the same exclusive UUID admission protocol as
workers, with distinct ownership retained until CJ removal is confirmed. Worker
GPUs remain task-scoped and can be reused while CJ stays alive. Both allocations
need sufficient free devices in the approved partition. Keep both images on the
same reviewed NVFlare revision.

## GPU admission profile

Add an approved partition to the client's `task_worker_config` to enable GPUs:

```json
{
  "gpu_partition": {
    "id": "site-approved-exclusive-pool",
    "exclusive": true,
    "devices": ["GPU-12345678-1234-1234-1234-123456789abc"]
  }
}
```

Use the actual full physical GPU UUIDs recorded on the daemon host. The example
UUID is a placeholder. GPU ordinals, `all`, duplicate UUIDs and MIG identifiers
are rejected. An administrator must approve these devices exclusively for this
admission protocol and exclude Process jobs, other schedulers, other daemons
and ad-hoc host workloads from them. This is an explicit partition profile,
not an allocator for arbitrary uncoordinated host GPU use.

Every participating CP reserves each device through an atomic Docker container
name keyed by its UUID. These small **never-started containers** are ownership
records, not worker/guardian processes, and have no GPU device requests.
Name conflicts prevent double assignment across CPs/jobs on the same daemon,
including overlapping partition configurations. A task cannot start until all
requested devices are reserved. Failed partial admission releases only its own
records; there is no automatic physical retry or scheduling queue.

Worker containers receive UUID-specific NVIDIA device requests and matching
visibility. The container's CUDA ordinals are local to that admitted set.
The NVIDIA driver and NVIDIA Container Toolkit must be installed and verified
on the host, and the worker image must contain the application/CUDA dependencies.
CPU workers create no GPU reservations even when the site has a partition.
An independent positive CJ GPU budget still reserves its own devices.
Multiple exposed GPUs do not imply torchrun/DDP support.

The worker's `resources.json` records native reservation IDs, admitted UUIDs,
partition, CPU/memory limits and attempt identity. Ownership labels and native IDs are verified
before controlling a reservation. Worker removal must be confirmed before
reservation removal; an unconfirmed response, lost owner record or changed
identity prevents successful settlement and fences further job admission.
CP crash/restart leaves records in place for operator reconciliation rather
than silently freeing capacity. Cleanup tools must preserve active reservation
records: do not run unrestricted Docker container pruning during execution.

On a site opting into a partition, resident jobs remain supported but cannot
request those devices or ambiguous count/all/ordinal GPU access. Their GPU
access must use explicit UUIDs outside the partition. Sites without a partition
keep existing resident JobLauncher GPU behavior.

## Reproduction and required deployment evidence

Run the focused tests from a checkout with development dependencies installed:

```bash
python -m pytest -q \
  tests/unit_test/app_opt/job_launcher/docker_launcher_test.py \
  tests/unit_test/app_opt/task_launcher/docker_launcher_test.py \
  tests/unit_test/app_opt/task_launcher/docker_authority_test.py \
  tests/unit_test/private/fed/app/client/task_worker_container_test.py \
  tests/unit_test/private/fed/client/task_execution_test.py \
  tests/unit_test/private/fed/task_worker
```

Use the existing `NPTrainer` training contract for initial deployment checks:
small NumPy weights, `train` only and two assignments. It already has Process
task-worker regression coverage and needs no recipe changes. Do not add local
model-submission/evaluation paths to this lifecycle smoke test. Export the same
application for resident and task runs with `FedJob.set_execution_lifetime`.
For example, this exports a matched pair without changing a trainer or recipe:

```python
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.job_config.api import FedJob

for lifetime in ("job", "task"):
    job = FedJob(name=f"docker-cpu-{lifetime}", min_clients=2, meta_props={
        "launcher_spec": {"default": {"docker": {"image": "site-approved-worker:reviewed-source"}}}
    })
    job.set_execution_lifetime(lifetime)
    job.to_server(NPModelPersistor(model=[[1.0, 2.0]]), id="persistor")
    job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator")
    job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    job.to_server(ScatterAndGather(min_clients=2, num_rounds=2, persistor_id="persistor"))
    job.to_clients(NPTrainer(delta=1), tasks=["train"])
    job.export_job("/tmp/docker-task-jobs")
```

Start the prepared Docker server/client kits with their normal
`startup/start_docker.sh`; submit the exported jobs through the admin client.
Site configuration supplies the Docker authority and storage bindings.

Record the source revision, image digests, daemon ID/version, UID/GID, job/site/
task/attempt/container IDs and the resolved host-to-container mounts. Required
checks before claiming a deployed CPU/GPU profile:

* Two assignments use different task container IDs, with the same CPU-only CJ.
  Compare numeric results to the resident run of the same application.
* Observe container exit, captured `container.log`, removal and an inspect
  response confirming absence before CJ's `settled`/`validated` diagnostics.
  Read and upload the committed result after removal. Delay publication and
  start the next CPU task while the previous immutable result is retained.
* Abort during execution and around completion; fail the worker before completion;
  and interrupt a Docker response. Confirm no success publication for an aborted
  attempt and no resource-release claim when settlement remains unconfirmed.
* Confirm a lost result receipt resubmits the saved result, without another
  container. Check receipt/closure payload cleanup, explicit retention, and
  active-reader survival. Preserve failed/incomplete/unresolved evidence.
* Run resident Docker JobLauncher regressions and a resident deployed job.
  Repeat task/resident checks under secure ProdEnv with job-scoped mTLS.
  Task evidence must show pre-compute HELLO and live metric acknowledgement;
  unreachable CJ DNS/listener, missing credentials and another job's certificate
  must fail before compute. Local Cell/mTLS tests do not qualify Docker DNS or
  mounted credential paths on the deployment host.

GPU qualification additionally requires real CUDA computation in an authorized
GPU image, CPU-only CJ verification, contention between independent jobs/CPs
using the same UUID, failure/cancellation release, and a second admitted task
reusing that GPU before a delayed first upload finishes. Verify no owned worker
or reservation records remain after supported teardown. Record inventory and
actual memory/process reuse evidence; passing CPU/mocked tests is insufficient.
Abrupt CP failure/restart adoption remains outside this slice.

The backend-owned probe exports a one-client, two-assignment job and asserts
real device count and computation. Use the same CUDA-capable image with this
source and PyTorch for both CPU and GPU runs:

```bash
PYTHONPATH="$PWD" python tests/integration_test/data/docker_task_execution/export_probe.py \
  --image site-approved-worker:reviewed-source --device cpu --clients 1 \
  --lifetime task --output /tmp/docker-task-probes
PYTHONPATH="$PWD" python tests/integration_test/data/docker_task_execution/export_probe.py \
  --image site-approved-worker:reviewed-source --device cuda --gpu-count 1 --clients 1 \
  --lifetime task --output /tmp/docker-task-probes
```

The GPU run must record nonzero CUDA allocation and kernel checksums in the
`docker_task_compute` log records, with the allocated UUIDs matching the CP
resource manifest. The CPU run asserts zero visible GPUs. Server requests are
explicitly CPU-only. `--lifetime job --device cpu` exports the resident CPU
comparison. A resident GPU comparison needs its own approved non-overlapping
partition or an isolated campaign after task resources are fully settled;
do not bypass the task partition guard to run it concurrently.

NVIDIA's [GPU enumeration documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
describes device visibility; the [Docker Engine API](https://docs.docker.com/reference/api/engine/version/v1.51/)
provides named container creation/removal. Neither visibility alone nor ordinary
job-level device requests provide the admission protocol described here.
