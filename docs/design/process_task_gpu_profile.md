# Process task GPU visibility

Process task execution accepts an explicit site-owned GPU assignment and applies
it to disposable workers through CUDA_VISIBLE_DEVICES. The launcher does not
schedule GPUs, reserve capacity, track memory budgets, exclude other processes,
or guarantee free memory or a compute share. Device placement and contention
policy belong to the site administrator or an external scheduler.

## Configuration and responsibility

Configure the canonical site local/resources.json (or a supported HOCON/YAML or
default variant), retaining existing components and scheduling configuration:

```json
{
  "task_execution": {
    "launcher": {
      "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
      "args": {
        "gpu_devices": ["GPU-00000000-0000-0000-0000-000000000001"]
      }
    },
    "worker_timeout": 300,
    "settlement_timeout": 10
  }
}
```

The ordered list accepts CUDA device ordinals, GPU UUIDs or MIG device IDs. Prefer
UUIDs for stable identification. A task requesting N GPUs receives the first N
configured IDs. CPU tasks receive an empty CUDA visibility list. Requests larger
than the configured list are rejected before spawn. The launcher does not probe
utilization, choose an idle device or validate actual CUDA availability; the
workload reports CUDA initialization or execution failures normally.

This is a static assignment. Two simultaneous CJs with the same configuration
can use the same GPU. Administrators must assign devices or accept sharing;
NVFlare supplies no local queue or GPU inventory accounting. A MIG identifier
selects an existing platform-configured instance; NVFlare does not create one.

Applications select execution_lifetime: "task". Task counts use job metadata:

```json
{
  "launcher_spec": {
    "default": {"process": {"num_of_gpus": 1}}
  }
}
```

The Process task profile rejects job-level resource_spec GPU requests, which
would use the existing job admission path, and rejects other backend GPU requests.
The opted-in client JobLauncher hides GPUs from resident CJs after study
configuration overrides. Only task workers receive the configured assignment.
Existing ResourceManager implementations and generic site inventories are
unchanged. Job and task launch backends must match.

Omitting gpu_devices preserves the CPU-only default task policy and ordinary
job-mode configuration. An explicit empty list enables an empty task assignment.
The earlier unreleased gpu_allocator/gpu_partition settings are removed. Stop and
settle owned workers before changing configuration; preserve any old ledger and
failure artifacts as historical evidence. There is no replacement ledger or
automatic modification of existing files.

## Failure handling and recovery

Each task executes in a fresh POSIX process group. Its handle retains ownership
of the leader until all owned descendants have settled. Leader exit by itself is
insufficient. Every descendant must remain in that group; detached processes,
external reaping and hostile code are outside the Process contract.

An uncaught CUDA OOM or other executor exception records a failed attempt and
runs executor finalization. The supervisor waits for or cancels the owned group
before returning the failure. Abrupt exit without a failure record still produces
a failed attempt when no valid completion exists. Timeout and cancellation also
clean up the whole owned group. Error and diagnostic artifacts remain available.

After confirmed cleanup, the supervisor can accept a fresh task with a new
attempt identity. A failed attempt is not silently rerun. Saved successful results
can be retransmitted without starting another worker. Worker execution and cleanup
finish before result filtering, publication or receipt handling, so a slow upload
does not keep a CUDA worker alive.

If process settlement cannot be confirmed, the supervisor retains its handle,
reports the uncertainty and rejects new work. It must not claim recovery while an
owned process may still be running. This fence is execution ownership protection,
not a GPU reservation. Operator investigation is required for unresolved cleanup.
A task that hangs should have a finite worker_timeout; cleanup timing still
assumes the OS can terminate and observe the owned process group.

Workers connect to the CJ through a live Cell session before computation. The
same-host Process topology uses the default localhost listener and the CJ's
job-scoped credentials, including internal mTLS when configured. Metrics travel
live; input, result, checkpoint and failure artifacts remain on disk. Each attempt
closes its session before the CJ releases its execution gate, including on OOM.

CP's automatic resident-CJ recovery cannot infer task worker settlement from CJ
process-group exit: disposable workers have their own groups. Once a task runtime
has been created, the existing runtime directory fences automatic CJ replacement.
After abrupt CJ death, an operator must confirm worker cleanup before restarting.
This does not prevent a living CJ from accepting fresh tasks after settled OOM or
other worker failure. No additional reservation or restart-state file is created.

Ordinary application OOM recovery comes from ending the worker and running the
next task in a fresh process. There is no GPU reset, driver repair, unrelated
process termination or device reconfiguration. Driver faults, unkillable processes
and abrupt CJ death can require operator intervention. A launcher cannot guarantee
that a shared host GPU is healthy or uncontended after its own worker exits.

## Qualification

The supported Process topology is a host CP, host CJ and disposable host worker
running trusted code. GPU simulator scheduling is not integrated. CPU/memory
resource enforcement and multi-GPU/DDP training qualification remain outside this
profile. The shared worker entry requires an explicit parent PID for Process
execution; native launchers use their own execution scope.

Follow the [qualification fixture](../../tests/integration_test/data/process_gpu/README.md)
for CPU/GPU job/task controls, real CUDA OOM and healthy follow-up work, descendant
cleanup, and delayed CJ publication with receipt retry. Tests of visibility alone
do not establish actual CUDA execution or hardware isolation.
