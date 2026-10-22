# Process task GPU profile

This profile keeps Process CP and resident CJs on CPU and assigns
whole NVIDIA GPU identifiers to disposable workers through a configured allocator.
It uses the existing TaskLauncherSpec, TaskHandleSpec, TaskLaunchRequest, task supervisor,
artifact completion and result-receipt contracts. No additional federation or
CP-to-CJ task protocol is introduced.

## Authority and configuration

The administrator supplies a device inventory for Process tasks. All job
allocators and external workloads must exclude those devices by site administration.
Participating CJs on the same host use the same persistent local reservation directory and
site UID. Disjoint hosts or device partitions may use separate directories.
Different directories must never account for overlapping devices.

The Process launcher accepts a `GpuAllocatorSpec` component and applies its
assigned devices to the worker environment. The supplied `GpuReservationLedger`
implementation records device claims in a persistent SQLite ledger shared by
CJs. It coordinates participating launchers; it does not partition GPU compute
or memory, guarantee available capacity, detect outside users, or enforce device
access. Workers are trusted to obey their CUDA visibility assignment. There is
no `exclusive` switch: configuration alone cannot establish hardware exclusivity.

The existing CP ResourceManager continues to own job-lifetime reservations.
GPUResourceManager accounts GPU memory within one CP process. It is not used
as the task allocator or instantiated in each CJ to create independent inventories.
Both paths use GPU utilities for discovery and CUDA visibility; their accounting
lifetimes remain distinct. Job admission expires unallocated reservation tokens;
the task ledger retains admitted claims until process settlement is proved.

The live CP resource manager must declare zero job GPU capacity through
`ResourceManagerSpec.get_gpu_capacity()`. An empty `ListResourceManager`, an empty
`GPUResourceManager`, or a custom manager explicitly declaring zero is supported.
Unknown or positive capacity rejects CJ launch. The check uses the live manager,
so editing a site file cannot conceal a stale running CP. Sharing devices between
job and task allocators would require a common authority and is outside this profile.

Configure each site's canonical `local/resources.json`, retaining its other
settings and components. CONF/YAML variants and default files use the same
configuration resolution in the CP and CJ:

```json
{
  "components": [
    {
      "id": "resource_manager",
      "path": "nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager",
      "args": {"resources": {}}
    }
  ],
  "task_execution": {
    "launcher": {
      "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
      "args": {
        "gpu_allocator": {
          "path": "nvflare.app_common.resource_managers.gpu_reservation_ledger.GpuReservationLedger",
          "args": {
            "pool_id": "site-task-pool",
            "devices": ["GPU-00000000-0000-0000-0000-000000000001"],
            "reservation_root": "/var/lib/nvflare/process-task-gpus"
          }
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

The ledger retains its original `process-gpu.sqlite3` filename and schema.
When upgrading the previous `gpu_partition` configuration, configure
the allocator with the same `reservation_root` and device inventory, rename `id`
to `pool_id`, and remove `exclusive`. Never change the directory or delete the
ledger to bypass existing claims. The Process launcher no longer accepts
`gpu_partition` or constructs a hidden allocator from that dictionary.

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
reserve GPUs for the standing job. The site rejects mismatched launcher modes;
the client JobLauncher checks the live CP's declared job GPU capacity before
spawning CJs.
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
The shared worker entry requires an explicit owning `--parent_pid` greater than
one for Process execution and rejects a changed parent before compute. Native
launchers select `--execution_scope native` and do not use a local CJ PID.

The ledger rejects an admission request when the configured inventory is fully
claimed. It does not implement a resource queue. Native queue/startup timeout
settings do not apply to this Process profile. The common `worker_timeout`
remains the overall attempt budget.

Abrupt CJ death may leave workers or descendants running. Durable reservations
remain held, so another participating CJ cannot reuse those devices. There is
no automatic orphan cleanup, adoption, reservation expiry or replay after restart.
An administrator must confirm whole execution settlement before repairing stale
records. Retain the ledger and diagnostics; never erase the directory to unblock
a pending workload.

Opting out preserves existing job-mode and CPU task behavior. This task-allocation
profile cannot run a GPU resident job concurrently. To run a GPU job control,
first confirm all owned workers and allocations are settled, stop only owned
idle participants, remove the opt-in task allocator, and configure the normal
job GPU manager/consumer with disjoint per-site device capacity. Restart those
participants and run the resident control serially. Merely submitting it to an
empty job GPU pool is a failed control, not a compatibility pass.

Qualification setup and evidence requirements are in
[the Process GPU test fixture](../../tests/integration_test/data/process_gpu/README.md).
