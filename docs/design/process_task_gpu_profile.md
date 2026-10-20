# Process task GPU profile

This experimental profile keeps Process CP and resident CJs on CPU and gives
each disposable worker an exclusive allocation of whole NVIDIA GPUs. It uses
the existing TaskLauncherSpec, TaskHandleSpec, TaskLaunchRequest, task supervisor,
artifact completion and result-receipt contracts. No additional federation or
CP-to-CJ task protocol is introduced.

## Authority and configuration

The administrator dedicates a physical GPU partition to Process tasks. All job
allocators and external workloads must exclude that partition. Participating
CJs on the same host use the same persistent local reservation directory and
site UID. Disjoint hosts or device partitions may use separate directories.
Different directories must never account for overlapping devices.

The existing CP ResourceManager continues to own job-lifetime reservations.
GPUResourceManager accounts GPU memory within one CP process, and its consumer
sets that process's environment. It is therefore not a shared task-attempt
authority. This initial profile requires its job GPU pool to be empty. Sharing
devices between job-lifetime and task allocations would require integrating a
common CP/host authority and is outside this profile.

Configure each site's canonical `local/resources.json`, retaining its other
settings and components:

```json
{
  "components": [
    {
      "id": "resource_manager",
      "path": "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager",
      "args": {"num_of_gpus": 0, "mem_per_gpu_in_GiB": 0}
    }
  ],
  "task_execution": {
    "launcher": {
      "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
      "args": {
        "gpu_partition": {
          "id": "site-task-pool",
          "devices": ["GPU-00000000-0000-0000-0000-000000000001"],
          "exclusive": true,
          "reservation_root": "/var/lib/nvflare/process-task-gpus"
        }
      }
    }
  }
}
```

Replace the illustrative UUID with the administrator-approved UUID from
`nvidia-smi --query-gpu=uuid,mig.mode.current --format=csv,noheader`. The directory
must be owned by the site UID with mode 0700. Use a persistent local filesystem
with SQLite atomic transactions and locking. Network filesystems, deleting or
replacing the ledger, disk corruption and transparent storage recovery are
unsupported. Keep this authority outside job workspace cleanup.

Application client configuration still selects `execution_lifetime: "task"`.
GPU counts are declared in job metadata, for example:

```json
{
  "launcher_spec": {
    "default": {"process": {"num_of_gpus": 1}}
  }
}
```

Positive portable `resource_spec.num_of_gpus` is rejected because it would
reserve GPUs for the standing job. The site rejects mismatched launcher modes
and GPU pools for CJs. The client JobLauncher checks the live CP manager too,
so editing a site file cannot mask a running CP with a stale GPU pool.
ProcessJobLauncher applies the trusted CPU environment
after study environment overrides, before spawning the CJ. GPU workers receive
only their assigned full UUIDs in CUDA_VISIBLE_DEVICES. CPU task workers receive
an empty list. Job-controlled environment forwarding cannot select GPUs.

## Admission, settlement and result lifetime

SQLite serializes whole-attempt admission across CJs. An attempt identity is
recorded permanently, including after release, to prevent a restarted launcher
from executing it again. Allocations never expire on a timeout or owner death.
Insufficient capacity rejects launch without starting a worker or reserving a
partial allocation. This profile does not implement a task admission queue.

One handle owns the existing POSIX process-group handle and its GPU reservation.
It reports settled only after the whole owned group settles and the reservation
release commits. A durable admission fence precedes that transaction; lost commit
confirmation retains the fence even if allocation deletion took effect. Other
CJs cannot treat those absent rows as free capacity. Leader exit alone does not free a GPU. Failed release prevents
successful settlement, retains ownership/evidence, and fences that handle. The
supervisor retains its existing permanent fence on uncertain settlement.

Resource release precedes returning the result to the CJ. It does not wait for
result filtering, upload, receipt or artifact cleanup. Re-reading a saved result
does not reacquire resources or launch a worker. The independent shared reader
fix retains payloads while saved-result readers are active.

The ledger retains attempt identity, partition, device UUIDs, allocation/release
times, owner PID and physical execution ID. The existing task diagnostics and
completion records retain the application result and process facts separately.

## Supported boundaries

GPU execution requires Linux, NVIDIA CUDA, `nvidia-smi`, and whole physical GPU
UUIDs with MIG disabled. Counts from one through the partition size are admitted
atomically; CPU and memory requests remain unsupported. Device accounting and
visibility do not make an application use multiple GPUs or implement DDP. The
provided GPU qualification fixture uses one GPU per worker; multi-GPU accounting
unit coverage does not qualify multi-GPU training.

Every descendant must stay in the worker's POSIX process group. The handle must
be the leader's sole reaper. Process mode shares the site UID and filesystem and
is not a hostile-code sandbox or device-access enforcement boundary. Workers
must obey their CUDA assignment and may not detach processes or change groups.
This profile does not extend to a container CP or a separately managed scheduler.
GPU simulator scheduling is not integrated with this provisioned-site profile.

Abrupt CJ death may leave workers or descendants running. Durable reservations
remain held, so another participating CJ cannot reuse those devices. There is
no automatic orphan cleanup, adoption, reservation expiry or replay after restart.
An administrator must confirm whole execution settlement before repairing stale
records. Retain the ledger and diagnostics; never erase the directory to unblock
a pending workload.

Opting out preserves existing job-mode and CPU task behavior. This task-exclusive
profile cannot run a GPU resident job concurrently. To run a GPU job control,
first confirm all owned workers and allocations are settled, stop only owned
idle participants, remove the opt-in task partition, and configure the normal
job GPU manager/consumer with disjoint per-site device capacity. Restart those
participants and run the resident control serially. Merely submitting it to an
empty job GPU pool is a failed control, not a compatibility pass.

Qualification setup and evidence requirements are in
[the Process GPU test fixture](../../tests/integration_test/data/process_gpu/README.md).
