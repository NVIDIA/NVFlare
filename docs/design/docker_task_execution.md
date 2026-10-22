# Docker task execution: CPU and GPU contract

For the setup sequence, image checks, and reuse of an existing host after a
source fix, see [Docker task worker setup and test procedure](../user_guide/admin_guide/deployment/docker_task_execution.rst).

Base: upstream `1b5e7f9832e6a777c54e22c72e6631bbd344babe`, including
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
* **Storage:** single-daemon Linux bind mounts. The persistent daemon-host job
  directory appears at `/var/tmp/nvflare/workspace/<job_id>` in both CJ and worker.
  Bootstrap paths and opaque checkpoint records retain their existing format.
  Startup and site configuration are read-only mounts; only this job directory
  is writable. Dataset bindings come from site-owned study configuration. The
  same numeric UID/GID as CP preserves mode-0600 artifacts. Different worker
  workspace paths, remote artifact transfer and ephemeral-only storage are
  outside this first profile. Host source paths are never inferred from CJ paths.
* **Resources:** CPU or GPU workers, with Docker CPU/memory limits supplied by site policy.
  Each job session admits at most one unsettled execution. These limits do not
  reserve an exclusive share of the host CPU pool. Worker containers have no
  network; this initial profile uses local mounted data.
  CJ always has GPU visibility disabled and no device requests. CPU workers
  likewise have no GPU access. GPU workers receive explicit physical UUIDs from
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

## Shared integration boundary

The Docker owner supplies the following shared changes, agreed with the
Kubernetes and Slurm owners:

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
START_RUN binding owns the native resources; Docker also carries the CP-bound
GPU count in the existing TaskResourceRequest. Positive whole-job CP GPU
reservations are rejected until portable resource routing can delegate them.

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
can access that job-scoped credential. Clearing credential environment variables
and disabling worker networking do not isolate this file. This profile therefore
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
before. Worker image and command policy remain site-owned. GPU requests resolve
from this site's Docker-native `launcher_spec[site]["docker"]["num_of_gpus"]`,
with the existing default/site precedence and explicit zero override. Positive
portable or legacy ResourceManager GPU requests are rejected: the current CP
ResourceManager holds those reservations for the whole job. Portable CPU/memory
fields remain supported. CP independently binds the native GPU count to the
job's session; CJ cannot resize its allocation through RPC. GPU jobs without an
approved partition fail before CJ creation. CPU/memory limits
in job metadata retain their resident-CJ meaning; worker limits above are
independent site policy. Keep both images on the same reviewed NVFlare revision.

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
CPU jobs create no GPU reservations even when the site has a partition.
Multiple exposed GPUs do not imply torchrun/DDP support.

The worker's `resources.json` records native reservation IDs, admitted UUIDs,
partition and attempt identity. Ownership labels and native IDs are verified
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
