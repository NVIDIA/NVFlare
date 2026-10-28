# Process task GPU visibility

Process task execution accepts an explicit site-owned GPU assignment and applies
it to disposable workers through CUDA_VISIBLE_DEVICES. The launcher does not
schedule GPUs, reserve capacity, track memory budgets, exclude other processes,
or guarantee free memory or a compute share. Device placement and contention
policy belong to the site administrator or an external scheduler.

## Choose the execution lifetime

The application chooses job or task lifetime; the site chooses the launcher.
`in_process` describes where the Client API runs relative to its executor. In
task lifetime that executor is inside the disposable worker, not inside CJ.
Process executes an ordinary Python task script directly in that worker. The
Kubernetes implementation and Slurm task-launcher work use the same worker
contract within their native execution scope, without a local Popen trainer.

| Lifetime / Client API mode | Where the script runs | Current support |
| --- | --- | --- |
| `job` / `in_process` | Inside the resident CJ. | Existing behavior; preserves in-memory state between assignments. |
| `job` / `external_process`, `launch_once=True` | An NVFlare-owned trainer kept across assignments. | Existing behavior. |
| `job` / `external_process`, `launch_once=False` | A fresh NVFlare-owned trainer for each task. | Existing behavior; does not select task lifetime. |
| `job` / `attach` | An externally managed trainer connected to CJ. | Existing behavior; the trainer's lifetime remains externally owned. |
| `task` / `in_process` | Directly inside the disposable worker selected by the site. | Supported Python script path; Process GPU qualification is described below. |
| `task` / `external_process` or `attach` | No launch. | Rejected by shared export and worker validation for every backend. |

Job lifetime already offers a fresh interpreter through `external_process` with
`launch_once=False`. Task lifetime adds the shared durable handoff: validate the
staged result, confirm settlement of the entire owned compute scope, and only then
publish through CJ. Saved-result retransmission needs no live trainer. Local
completion, server receipt and algorithm acceptance remain separate facts. A
job-lifetime external trainer can remain alive while its lazy result is transferred;
it does not acquire the task-lifetime settlement-before-publication guarantee.

Fresh workers repeat interpreter/library startup, CUDA initialization, model
construction and parameter loading on each assignment. Optimizer state must be
recreated or explicitly checkpointed and restored; it is not retained in memory.
Staging output also adds I/O. Prefer resident job mode when keeping a large model
or optimizer in memory, reusing an expensive environment, or maintaining a live
session matters more than releasing compute between tasks.

For an ordinary Python Client API script, use the existing recipe defaults and
select only the lifetime:

```python
import torch
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="process-task",
    model=torch.nn.Linear(2, 1),
    min_clients=2,
    train_script="train.py",
    execution_lifetime="task",
)
recipe.export("/tmp/process-task-jobs")
```

The recipe's default `in_process` ScriptRunner packages the script for direct
worker execution. `FedJob(execution_lifetime="task")` with a default ScriptRunner
is the equivalent lower-level export path. GPU counts and site device visibility
are configured separately below. Do not add `launch_external_process=True`, an
external command, or an internal worker executor for this path. Unsupported
external commands are rejected; they are never converted into scripts or silently
changed to job lifetime.

Arbitrary commands, separate Python environments and `torchrun` in task lifetime
are a shared parity gap tracked under [#5434](https://github.com/NVIDIA/NVFlare/issues/5434).
A follow-up must launch the command from the shared task worker within the site's
owned execution scope, inherit its allocation/environment, cancel its process tree,
stage durable output and confirm full settlement before CJ publishes. Process,
Kubernetes and Slurm need the same contract. Removing the current guards or adding
a Process-only command launcher would not supply those guarantees. Attach remains
externally owned/resident unless a separate ownership contract is designed. Docker
task support is deferred; existing Docker job support is unchanged.

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
  "resource_spec": {
    "@default": {"num_of_gpus": 1},
    "site-2": {"num_of_gpus": 0}
  }
}
```

Submission and export derive task sites from the actual client configurations
and add an independent cj_resource_spec entry for each task site. For these sites,
resource_spec describes disposable compute; it does not reserve that GPU count
for the CJ lifetime. In this example site-2 explicitly requests a CPU task while
other task sites request one GPU. A site count of zero overrides the default.

Process CJs use the host's ordinary process resources and hide GPUs after study
configuration overrides. Explicit CJ CPU/memory budgets and positive CJ GPU
counts are rejected because this launcher cannot enforce them. Only task workers
receive the configured device assignment. There are no implicit Kubernetes CJ
resource defaults in Process mode. Older task metadata that would reserve compute
GPUs through whole-job admission must be resubmitted through normalization.

The legacy launcher_spec.default.process.num_of_gpus declaration is also
supported. If both compute resource_spec and Process launcher_spec specify a GPU
count, they must agree. Other backend GPU declarations, compute CPU/memory
budgets, per-GPU memory quotas and custom resource claims are rejected. Existing
ResourceManager implementations and generic site inventories are unchanged;
job-mode admission keeps its existing meaning. Job and task launch backends must
match.

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
