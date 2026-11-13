Slurm task execution
====================

This guide is for site operators enabling disposable CPU or GPU workers.
Start with :ref:`slurm_job_launcher` for provisioning, parent placement,
networking and ordinary job-lifetime execution. Both lifetimes use
``nvflare.app_opt.job_launcher.slurm.launcher.ClientSlurmJobLauncher``.
The deployed application's ``execution_lifetime`` selects the behavior:

* ``job`` runs computation in the resident Client Job (CJ) allocation.
* ``task`` keeps CJ for federation coordination and creates a separate Slurm
  allocation for each task attempt. The persistent Client Parent (CP) submits
  the workers and retains responsibility for their cleanup.

Applications must save and load optimizer, RNG and data-position state needed
across workers. The implementation uses the common task runtime. For design
and lifecycle details, see the
`Slurm launcher design <../../../design/slurm_job_launcher_design.md>`_.

CPU and GPU task profiles
-------------------------

Configure the resources each application needs; a site can run CPU-only tasks
without a GPU task profile. The task request selects resources independently
of the CJ: ``gpu_count=0`` submits ``--gres=none``; a positive count submits
``--gres=gpu:<count>`` and receives device visibility from that allocation.
Explicitly clearing GRES is limited to this task-capable profile. Existing
resident ``ClientSlurmJobLauncher``/``ServerSlurmJobLauncher`` behavior is
unchanged: portable zero GPU metadata adds no GRES option, and native
``gpus_per_node: 0`` remains invalid. Clusters without a GRES plugin have not
been qualified for the experimental task profile's ``--gres=none`` option.
CPU/memory requests, cancellation, immutable result handoff, receipt retention
and cleanup observation apply to both. Normal worker teardown releases its
allocation before result upload; unconfirmed release is reported separately
from the successful application result.

Native Slurm GPU admission does not depend on the Process backend's GPU
allocator. The existing CPU-only Process implementation is a base-revision
limitation, not the target resource scope for either backend. Process GPU work
is separate from this Slurm implementation. Supporting a GPU resource request
does not add multi-machine execution or a torchrun/DDP contract.

CP placement
~~~~~~~~~~~~

Run CP as a persistent service outside Slurm when possible. It must have
scheduler access and access to the shared workspace, and its configured
``parent_host`` must be reachable from CJ nodes. CP may also run inside a
Slurm allocation; ``SLURMD_NODENAME`` supplies the parent host when no explicit
``parent_host`` is configured. In either placement CP must outlive its CJs
and task cleanup. A CP allocation wall limit can interrupt that ownership;
this adapter does not recover owners after CP itself crashes.

CJ and compute resource budgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing portable ``resource_spec`` and native Slurm ``launcher_spec`` fields
describe compute. With job lifetime they apply to the resident CJ. With task
lifetime they apply to each physical task allocation. A task-mode CJ uses
the separate ``cj_resource_spec`` budget instead of inheriting compute CPU,
memory or GPUs. Omitted CJ fields use the trusted site's ``cj_cpu_cores``
(default 1), ``cj_memory_mb`` (default 1024) and ``cj_gpu_count`` (default 0).
An explicitly requested CJ GPU is independent of task GPUs and remains held
for that CJ's lifetime. The site's CJ partition must support that request.

For example, a task application may supply::

    {
      "resource_spec": {
        "@default": {"num_of_cpus": 4, "memory": "4Gi", "num_of_gpus": 1}
      },
      "cj_resource_spec": {
        "@default": {"num_of_cpus": 1, "memory": "1Gi", "num_of_gpus": 0}
      }
    }

Shared export and server submission derive task sites from the actual client
configs, including mixed lifetimes and ``@ALL`` deployments. They add the
applicable CJ entries even when the application omits a budget. The shared
resolver merges ``@default``, ``@ALL`` and site-specific CJ fields; this
backend does not create another metadata schema or infer admission roles
from a supplied flag. CP and CJ validate the locally resolved lifetime
before application components start. Resident-job admission remains based
on its compute requirements.

Only the CJ budget enters long-lived admission. Compute CPU, memory and GPU
requests belong to each task's native allocation and are allocated by Slurm independently of the long-lived CJ budget. Every task allocation requests CPU and memory, including GPU
tasks; trusted task defaults are one CPU and 1024 MiB. Slurm allocates whole
CPU cores, so a fractional CJ CPU share rounds up to the next whole core.
Task CPU requests must be whole cores. Memory quantities convert to exact
MiB. This adapter rejects custom resource-manager requirements and positive
per-GPU memory requirements that it cannot enforce.

Native ``launcher_spec`` timing fields apply to both the CJ and its task
allocations: ``time`` is preserved for each allocation, and ``pending_timeout``
is bounded by the respective trusted site limit. Splitting CPU, memory and
GPU budgets does not discard the job timing policy.

Storage and recovery
---------------------

The current bare profile shares the site workspace under one UID. It requires
trusted job code and does not enforce CP-private storage, per-job filesystem
isolation or scheduler-credential isolation. The launcher warns at startup;
mTLS does not add filesystem isolation.

Keep required checkpoints and application state in the persistent job directory.
The common artifact store promotes opaque checkpoint state from verified
application completion. Unconfirmed native cleanup retains attempt payloads
and scheduler control records for investigation; it does not invalidate an
already verified result or checkpoint.

Resident ``job`` lifetime retains the common automatic CJ recovery policy.
``max_cj_restart_failures: 0`` disables it. The budget counts consecutive failed
starts and resets at READY, rather than being a lifetime or per-job cap.
Task-mode CJ recovery remains disabled: independent worker allocations can
outlive CJ. CP continues cleanup while alive; CP crash/restart adoption is not
implemented. Persistent files alone do not authorize automatic physical retry.

Live worker connections and credentials
---------------------------------------

Workers connect to their owning CJ's internal Cell listener before running
compute. The Slurm launcher advertises the CJ host from ``socket.getfqdn()``
by default. If that name is not routable from compute nodes, set the trusted
``task_execution.launcher.args.worker_listener_host`` to the CJ's reachable
host name or IP address. The listener uses its own dynamic port and inherits
the site's internal transport and connection security. Allow compute-node
connections to that host and port range. Wildcard addresses and URLs with
ports are rejected. Explicit loopback is suitable only for a qualification
deployment whose CJ and workers all share one host.

For mTLS, the bootstrap must contain a shared, readable root CA and both
client and server credential pairs. Both pairs must name this job's
``job_cert/job.crt`` and ``job_cert/job.key`` under its persistent run
directory. The launcher checks every path before submitting the worker;
workspace aliases are resolved before containment and job-path checks.
Private host paths, traversal outside the workspace, site-certificate
fallback and symlink credential files are rejected. A symlink used for the
workspace directory is permitted.
The worker joins the authenticated job Cell session and delivers analytics
live while the attempt is owned. Analytics are not spooled for post-exit
replay. Task inputs, results and checkpoints still use the shared artifact
store; worker diagnostic logs retain their separate collection path.

Shared filesystem and scheduler prerequisites
---------------------------------------------

The initial profile uses effective Slurm sandbox ``none``. CP, CPU CJ and
task allocations must share the same workspace and code/data paths. The
interpreter, Python dependencies and custom code must exist on the compute
host at the specified paths. Task cwd must be a canonical directory inside
the configured workspace. Local scratch, ephemeral allocation directories,
object-store mounts with different semantics and remapped containers are
outside this profile.

The artifact filesystem must support exclusive hardlink publication,
immutable mode-0600 files, atomic same-filesystem replacement and file and
directory ``fsync``. It must provide coherent visibility across the CJ and
compute hosts after publication. Checkpoints reuse the existing opaque
payload plus atomic manifest promotion; there is no additional state API.
Initialization exercises hardlinks, exclusive installation, replacement,
active-descriptor survival and fsync locally. This does **not** establish
cross-host coherence. Qualification must repeat these checks across hosts,
then read a real committed result and checkpoint after allocation teardown.

Slurm accounting and step records must be available. Task accounting checks
the submitter UID and attempt-derived job name; when the site stores job
comments (``AccountingStoreFlags=job_comment``), a different comment is also
rejected. Live queue and marker recovery still require the exact job comment.
Duplicate accounting identities are rejected rather than choosing a potentially
requeued or recycled job ID. Accounting permission, retention and update latency
must cover each task's full settlement window.

The site must enforce ``proctrack/cgroup`` and ``task/cgroup`` with the
required CPU/memory/device constraints. All worker descendants must remain
inside scheduler-owned cgroups. This is required for controller/accounting
evidence to establish whole-allocation settlement; a process-group-only
Slurm configuration is insufficient. ``shared_filesystem=True`` and
``cgroup_enforced=True`` are trusted operator attestations, not discovery or
changes to scheduler configuration. They must be checked against the actual
deployment before enabling this profile.

The controller reports ``COMPLETING`` while cleanup is in progress; the
adapter waits for disappearance and terminal accounting. See Slurm's
`job states <https://slurm.schedmd.com/job_state_codes.html>`_,
`accounting fields <https://slurm.schedmd.com/sacct.html>`_ and
`cgroup tracking <https://slurm.schedmd.com/cgroup_v2.html>`_.

Configure each client site
--------------------------

The site operator configures task execution in that client's prepared
workspace, not in the application job. After ``nvflare deploy prepare``, copy
``<client-workspace>/local/resources.json.default`` to
``<client-workspace>/local/resources.json`` and edit the copy. NVFlare reads
``resources.json`` when present, otherwise the ``.default`` file. Repeat for
each participating client site; paths, partitions and accounts can differ.
Running ``deploy prepare`` again regenerates the default and removes the
active copy, so reapply these site settings after redeployment.

Keep the generated ``ClientSlurmJobLauncher`` component; do not add another job
launcher. Its ``args`` control CJ placement and the default coordinator budget
(``cj_cpu_cores``, ``cj_memory_mb``, ``cj_gpu_count``). Add this top-level
``task_execution`` entry to the same file for task workers:

.. code-block:: json

    {
      "task_execution": {
        "settlement_timeout": 60,
        "launcher": {
          "path": "nvflare.app_opt.job_launcher.slurm.task_launcher.SlurmTaskLauncher",
          "args": {
            "workspace_path": "/shared/private/site-1",
            "shared_filesystem": true,
            "cgroup_enforced": true,
            "python_path": "/shared/env/bin/python",
            "cpu_cores": 4,
            "memory_mb": 4096,
            "gpu_count": 0,
            "sbatch_directives": {"account": "site-account", "partition": "task-compute"},
            "poll_interval": 10,
            "cancel_timeout": 120
          }
        }
      }
    }

Keep the rest of the generated ``resources.json`` content. The task launcher
must use the same ``workspace_path`` as the generated client job launcher;
its ``sbatch_directives`` can route workers to a different partition.

The CP reads these trusted site settings, submits the CJ allocation, and owns
submission and cleanup of each task allocation. CJ uses a job-scoped proxy to
request tasks from CP; it does not need ``sbatch`` permission. The same site
file is visible to CJ so it can validate the job's resource binding. The
application job selects ``execution_lifetime="task"`` and requests compute
resources with ``resource_spec`` or ``launcher_spec``; it may set
``cj_resource_spec`` for the coordinator. The job cannot select a launcher
class, scheduler executable or site account.

The existing Slurm parent-listener/network configuration still applies.
Scheduler executable paths, when configured, must be available on CP.
The existing Slurm resolver locates CLI tools on CP when paths are omitted.
The worker interpreter must be available on compute hosts.
The CJ adapter reads the deployed application lifetime and preserves ordinary
resident-job behavior. In task lifetime it submits a separate CJ request
using its coordinator CPU/memory/GPU budget and the preserved timing policy.
With the default zero-GPU CJ it submits
explicit ``--gres=none`` and masks GPU visibility. Original compute metadata
remains intact for task admission.

Worker defaults come from the task launcher. Existing native
``launcher_spec["default"]["slurm"]`` and site overrides then determine
``gpus_per_node``, ``cpus_per_node``, ``mem_per_node``, time and pending timeout.
For example, ``launcher_spec["site-1"]["slurm"]={"gpus_per_node":1}`` requests
one task GPU; explicit zero overrides a GPU default. Portable CPU and memory
fields retain their usual mapping and conflict checks.

Portable and legacy compute GPU counts use the shared compute resolver;
normalized task metadata keeps them out of the whole-job CJ reservation.
Old, unnormalized task metadata that would reserve compute GPUs for the full
job fails closed and must be resubmitted through updated export/submission.
Native Slurm GPU metadata remains supported. Task and CJ capability checks
reject requirements that the selected runtime cannot enforce.

Results, cancellation and cleanup
----------------------------------

The worker publishes immutable completion after successful application
finalization and checkpoint staging. CJ validates completion identity and the
result payload before accepting it. A successful result remains successful
when Slurm cleanup cannot be confirmed. Missing or corrupt completion and
application failures remain execution failures; explicit user abort is honored.

After receiving a verified result, CJ uses
``task_execution.settlement_timeout`` as its cleanup confirmation budget;
control requests have their own bounded query timeout. It then
acknowledges computation to CP, which requests cancellation if the allocation
is still unsettled and continues cleanup in the background. The next assignment
can proceed while CP retains the old allocation's identity and replay fence.
An acknowledgement does not mark native cleanup settled.
If a completion request or reply is lost, CJ retains the notification and
retries it before requesting the next allocation. These control requests use
the configured query timeout; an unavailable CP prevents a new submission
without changing the already verified result. Closing admission requests
cancellation for every allocation even if its notification was lost.

Unconfirmed cleanup produces a warning and diagnostic with the execution ID.
Elapsed time, missing accounting and successful ``scancel`` acknowledgement
never become release proof. Confirmed release uses controller absence and
matching terminal allocation/step accounting. Report result success and cleanup
confirmation separately when investigating a run or demonstrating GPU reuse.

The client job handle preserves CJ's outcome and limits its task-owner join to
``ClientSlurmJobLauncher.task_shutdown_timeout`` (default 30 seconds).
``SYSTEM_END`` uses the same bounded budget. CP logs unresolved identities and
retains cleanup ownership while alive. Administrative investigation and native
cancellation use the site's Slurm tools; there is no operator-release file or
NVFlare override that declares an allocation released.

After CP loss or restart, inspect its retained attempt inventory from the same
site workspace::

    python -m nvflare.app_opt.job_launcher.slurm.task_inventory --workspace /path/to/site/workspace

The read-only command emits one JSON line per retained attempt, including the
job, site, task, attempt, execution ID, scheduler job name/comment, and last
recorded state. ``unconfirmed`` includes submissions with no returned allocation
ID and incomplete or damaged journals. ``confirmed_settled`` means the former CP
recorded native settlement; it is not a fresh scheduler query. No entry is
adopted or cancelled by this command. Keep the task directories until the
corresponding allocations have been reconciled.
The inventory covers task allocations only. CJ and server-job allocations use
the resident Slurm job launcher and require separate scheduler reconciliation.

For each unconfirmed entry, check the current queue under the submitting UID
using its exact scheduler name and comment. If it has an allocation ID, inspect
that ID with ``squeue`` and ``sacct --duplicates`` (including batch and other
steps). An unresolved ID needs a name/comment search and may require the site's
Slurm administrator to investigate delayed or missing accounting. Cancel only
an allocation whose ownership matches the retained identity, then recheck
controller disappearance and terminal allocation/step records. A successful
``scancel`` response, an empty queue, or elapsed time alone does not confirm GPU
release. If Slurm or accounting is unavailable, retain the inventory and
escalate through the site's normal incident process; CP restart does not adopt
these attempts or guarantee shutdown during the outage.

Recognized definite ``sbatch`` rejections fail promptly with scheduler text.
Site/Lua explanations may accompany a known rejection, including messages
printed at error level. Known resource validation errors and local errors such
as an unreadable batch script settle without an allocation. Unknown rejection
reasons, conflicting diagnostics, returned allocation IDs, lost replies and
transport timeouts retain native cleanup ownership. CP searches for an exact
name, UID and ownership-comment match and cancels it when identified. A
foreign-cluster response is cleaned up in the named cluster and never adopted
for execution.

Verification
-------------

Use the `Slurm qualification fixture <../../../../tests/integration_test/data/slurm_task_execution/README.md>`_
for CPU/GPU job/task exports, commands and evidence requirements. Local mocks
and CPU process tests do not qualify deployed scheduler behavior. Record the
application result, cleanup confirmation and capacity reuse as separate
observations. A result can pass while cleanup remains unconfirmed; a claim of
GPU release still needs Slurm evidence or a successful independent CUDA reuse.
