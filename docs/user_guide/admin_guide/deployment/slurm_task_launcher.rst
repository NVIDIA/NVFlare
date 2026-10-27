Slurm disposable task allocations (experimental adapter)
========================================================

The Slurm task adapter supports **both CPU and GPU tasks** through
``TaskLauncherSpec``, using one bare, single-node allocation per physical
attempt. It reuses the existing Slurm
manager's workspace validation, exclusive control directories, submission
serialization, scheduler command transport and sbatch option construction.
Resident ``ClientSlurmJobLauncher`` and ``ServerSlurmJobLauncher`` behavior is
unchanged.

The application uses its normal ``execution_lifetime="task"`` flag. Trusted
site resources select the matching Slurm CP/CJ/task profile. This branch
includes shared selection, saved-result reader pins, disk-backed task payloads
and live worker Cell sessions from the common runtime. Local tests include canonical
site selection and the ordinary client configurator/executor path, but do not
qualify a deployed scheduler. The broader acceptance criteria in
`issue 5434 <https://github.com/NVIDIA/NVFlare/issues/5434>`_ remain open.

Backend contract
----------------

* **Authority:** a resident CPU CJ with site-authorized scheduler access
  submits its task allocation directly. The adapter requires the Slurm child
  marker and CJ allocation ID. Those checks do not establish the complete
  CP/CJ launch chain; trusted same-mode selection must establish it.
  The CPU CJ is the communication/supervision allocation; task computation
  may use CPUs or GPUs in its separate allocation.
* **Identity:** the complete job/site/task/attempt tuple has an exclusive,
  persistent control directory under ``.nvflare_slurm/tasks``. The scheduler
  name and comment carry its hash. The handle exposes allocation ID, local
  cluster identity and attempt hash. There is no requeue, physical retry,
  restart adoption, job-array, multi-cluster or multi-machine support.
* **Resources:** ``TaskResourceRequest`` supplies whole CPU cores, memory MiB
  and native Slurm GPU count independently of the CJ allocation. CPU-only
  requests explicitly submit ``--gres=none``. The trusted site owns account,
  partition, QoS, reservation, constraints, time limit and runtime setup.
  GPU visibility comes from the new allocation, never the CJ environment.
* **Storage:** the current bare adapter requires code, interpreter, data,
  bootstrap, inputs, results and opaque checkpoints at identical absolute
  paths. The same numeric UID owns the site workspace on every host. This
  does not enforce the private-CP/shared-jobs boundary described below.
  No copying of host-private paths, remote staging or container path mapping
  is implemented in this revision.
* **Settlement:** successful ``scancel`` and batch-leader exit do not settle
  an attempt. Healthy controller queries must report the allocation absent
  before and after identity-checked accounting. The allocation and every
  returned batch/extern/numeric step must be terminal; successful allocations
  require batch accounting. Unknown, completing, requeued and unavailable
  states retain ownership. Missing accounting never ages into release proof.
* **Retention:** the adapter retains its scripts, identity and scheduler
  observations. It does not remove application artifacts or promote state.
  The existing supervisor validates immutable completion, owns receipts and
  applies payload retention. An unresolved submission retains a cleanup
  handle and fences its attempt even if sbatch lost its response.

CPU and GPU task profiles
-------------------------

Both profiles are required. The task request selects resources independently
of the CJ: ``gpu_count=0`` submits ``--gres=none``; a positive count submits
``--gres=gpu:<count>`` and receives device visibility from that allocation.
CPU/memory requests, cancellation, immutable result handoff, receipt retention
and whole-allocation settlement apply to both. The CJ's result upload must not
hold a task GPU allocation.

Native Slurm GPU admission does not depend on the Process backend's GPU
allocator. The existing CPU-only Process implementation is a base-revision
limitation, not the target resource scope for either backend. Process GPU work
is separate from this Slurm implementation. Supporting a GPU resource request
does not add multi-machine execution or a torchrun/DDP contract.

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
requests belong to each task's native allocation and are held until physical
settlement. Every task allocation requests CPU and memory, including GPU
tasks; trusted task defaults are one CPU and 1024 MiB. Slurm allocates whole
CPU cores, so a fractional CJ CPU share rounds up to the next whole core.
Task CPU requests must be whole cores. Memory quantities convert to exact
MiB. This adapter rejects custom resource-manager requirements and positive
per-GPU memory requirements that it cannot enforce.

Storage layout, persistence and CJ resume
-----------------------------------------

The accepted default site profile has two stores: one private CP workspace
store and one shared jobs-state store. CP retains access to both. Each CJ and
its workers receive only that job's directory from the jobs-state store;
other jobs and CP's private workspace must remain inaccessible. These are
per-site stores with per-job directories, not a new store per attempt.
Site launch policy, scheduler credentials and private launch controls belong
to CP. Sharing job files does not authorize a CJ to submit, inspect or cancel
native allocations, or to receive broad runtime credentials.

Both resident and task lifetimes must preserve required checkpoints,
application recovery metadata and committed task artifacts in the job's
persistent directory. Applications must explicitly write their required
recovery state there; allocation-local scratch and files outside that store
are not preserved by this contract. Checkpoint and completion publication
continue to use the existing immutable payload and atomic manifest rules.
Retention must not depend on a CJ completing its shutdown upload. CP still
needs access to job state for result collection and cleanup after CJ loss.

Resident ``job`` lifetime retains the common runtime's automatic CJ recovery
after its Slurm handle confirms allocation exit. Task-mode CJs report
``can_restart=False``: independent task allocations can outlive their CJ,
so CJ allocation exit alone cannot authorize a replacement. Task-mode recovery
requires trusted cleanup ownership that proves every earlier task has settled.
Persistent files do not authorize task adoption. Preserve unresolved launch
evidence and verify physical settlement before operator cleanup; neither a
surviving directory nor a missing process proves an allocation can safely be
replaced. Receipt-driven payload cleanup remains separate from retention of
required checkpoints and recovery metadata.

The current bare adapter and its qualification fixture share the site
workspace under one UID. They do **not** enforce the accepted storage split,
per-job filesystem exposure or CP-only scheduler authority. Completing that
boundary requires the CP-owned launch and isolation revision; the layout
above is its required contract, not a capability established by this
revision's passing CPU/GPU tests. Do not expose the CP store to job code as
a shortcut to making paths visible.

Qualification of that revision must check the actual mounts and permissions
for CP, CJ and workers, verify that CP-private and other-job paths are
inaccessible, and confirm that committed checkpoints and metadata survive CJ
loss in both lifetimes without relying on shutdown upload. Existing normal
CPU/GPU results do not establish those storage or resume properties.

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
private host paths, site-certificate fallback and symlinks are rejected.
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

Slurm accounting and step records must be available. The site must already
store job comments (``AccountingStoreFlags=job_comment``) for exact accounting
ownership checks. Duplicate accounting identities are rejected rather than
choosing a potentially requeued or recycled job ID. Accounting permission,
retention and update latency must cover each task's full settlement window.

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

Trusted adapter configuration
-----------------------------

In the canonical site resources file, select the exact matching pair::

    {
      "components": [{
        "id": "job_launcher",
        "path": "nvflare.app_opt.job_launcher.slurm.task_job_launcher.ClientSlurmTaskJobLauncher",
        "args": {
          "workspace_path": "/shared/private/site-1",
          "sandbox": "none",
          "python_path": "/shared/env/bin/python",
          "executables": {},
          "cj_cpu_cores": 1,
          "cj_memory_mb": 1024,
          "cj_gpu_count": 0,
          "sbatch_directives": {"account": "site-account", "partition": "cpu"}
        }
      }],
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

The existing Slurm parent-listener/network configuration still applies.
Executable paths, when configured, must be available on the respective CP/CJ
hosts. Otherwise the existing Slurm resolver locates the CLI tools there.
The CJ adapter reads the deployed application lifetime and preserves ordinary
resident-job behavior. In task lifetime it submits a separate CJ request
using only its coordinator budget. With the default zero-GPU CJ it submits
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

Construction performs no scheduler calls. First launch initializes the
manager and probes local storage and accounting. The runtime provides an
staged bootstrap command and canonical shared cwd. At START_RUN the launcher
binds immutable job/site identity, full native resource values and trusted
runtime policy. It converts that staged command to the configured Slurm Python
worker module, discarding the CJ's interpreter, parent PID and environment.
Missing or mismatched binding fails before submission. ``--export=NIL`` prevents inheriting CJ credentials and scheduler
resource requests. Trusted setup may amend the explicitly supplied environment;
the final worker receives those named values and allocation-owned Slurm/GPU
values. Request values cannot override scheduler identity, GPU visibility or
federation/Client API bootstrap credentials. Job arguments are shell-quoted.
Existing trusted study account/partition/QoS, interpreter, setup and plain
environment overrides are resolved for the task. The bare profile rejects
study secrets and container/dataset mount mappings rather than silently
discarding them. Data already accessible under shared paths needs no staging.

Logs use the retained control directory's ``scheduler.jsonl`` plus the run
directory's ``slurm-<allocation>.out``. Scheduler records correlate with
``identity.json`` and the handle's execution ID. Ordinary worker artifact
diagnostics remain under the supervisor-owned attempt directory.

Cancellation and unresolved execution
-------------------------------------

Queued and running cancellation latch intent and send scancel only after a
healthy exact-ownership queue observation. A concurrent successful completion
cannot erase cancellation intent. Acknowledgement is followed by polling for
allocation/step release. Cancellation timeout raises ``TaskSettlementError``;
scripts, diagnostics and the handle remain owned. A delayed terminal state
can subsequently be observed without rerunning compute.

The task adapter defaults ``cancel_timeout`` to 120 seconds to allow the
scheduler's termination grace period and epilog/accounting delay. Sites should
set it above their configured KillWait plus cleanup delay. A settlement wait
queries immediately with at most one second between queries while accounting catches up;
ordinary polling retains the configured Slurm poll interval. Neither timeout
expiry nor a cancellation acknowledgement authorizes resource release.

If sbatch throws, times out, returns an invalid response or reports another
cluster, ``TaskLaunchError.handle`` retains the unresolved execution. The
adapter cannot safely cancel a missing/unsupported physical identity. An
operator must resolve the exact scheduler name/comment and confirm release
before manual retention cleanup. It never cancels other jobs by a broad name,
deletes launch evidence or retries the same attempt automatically.

Shared integration contract
---------------------------

The adapter uses the shared task contracts for site admission, resource
roles, authorization, immutable completion and checkpoint/receipt retention:

* Export validates resource shape while deferring receiving-site capability.
  Default-deny validation remains for unsupported backends; Process GPU work
  is separate.
* ``TaskExecutionPolicy.from_config`` lazily selects the exact supported
  class. ``from_workspace`` calls ``launcher.validate_site_config(site_config)``
  with the complete canonical site dictionary. Slurm verifies the exact CJ
  adapter and shared bare workspace; mixed profiles fail closed.
* Non-Process worker environments come from backend site policy. Slurm's
  ``task_resources`` common hook returns an empty request after admission;
  START_RUN binding preserves its complete native resources. The shared
  executor forwards that existing event rather than adding a lifecycle API.
* Slurm's thin worker entry point verifies its allocation marker and delegates
  to the common worker with ``--execution_scope native``. The common path owns
  authorization, initialize/execute/finalize, Client API context binding,
  inherited job-credential removal and bounded log flush/exit, without a local
  CJ PID requirement. TaskHandle/Supervisor/artifact/receipt/checkpoint schemas
  stay unchanged.
* The shared saved-result reader fix pins payloads while a replay reads them;
  receipt cleanup waits for the last reader without holding compute or a
  state lock across disk IO. Read failure retains evidence. Deterministic
  regressions cover the previously reproduced replay/receipt deletion race.

Reproduction and remaining qualification
----------------------------------------

From this checkout, using an environment with project test dependencies::

    python -m pytest tests/unit_test/app_opt/job_launcher/slurm_task_launcher_test.py tests/unit_test/app_opt/job_launcher/slurm_task_integration_test.py -q
    python -m pytest tests/unit_test/app_opt/job_launcher/slurm_*_test.py -q
    python -m pytest tests/unit_test/private/fed/task_worker -q
    ./runtest.sh -s --skip-install

Export the shared qualification application from the exact candidate checkout::

    PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cpu --clients 2 --lifetime job --output /private/tmp/slurm-probes
    PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cpu --clients 2 --lifetime task --output /private/tmp/slurm-probes
    PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cuda --gpu-count 1 --clients 2 --lifetime job --output /private/tmp/slurm-probes
    PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cuda --gpu-count 1 --clients 2 --lifetime task --output /private/tmp/slurm-probes

These commands only export jobs. Submit them through the normal site job
workflow on the authorized Slurm CP/CJ/worker deployment. The CPU/CUDA compute
script is byte-identical to the shared backend probe; only Slurm metadata is
adapted. CPU asserts zero visible CUDA devices; CUDA requires actual allocation,
kernels, reduction and synchronization (64 MiB and checksum 33554432 per
requested GPU by default). Record the script and candidate source hashes.
The historical ``docker_task_compute`` log-event name remains unchanged across
backends. Join these computation records with bootstrap/completion identities,
supervisor execution IDs and Slurm allocation/step/device evidence; PID or an
environment value alone does not identify an allocation or prove GPU ownership.

Repeat all four exports with ``--resource-format portable`` to exercise
``resource_spec["@default"]`` and the independent CJ budget through the actual
submission path. These jobs have a ``-portable`` name suffix. They deliberately
request different values from the native fixture and trusted task defaults:
two CPUs and 3 GiB per compute allocation, one CPU and 2 GiB for the server,
and, in task lifetime, two CPUs, 1536 MiB and zero GPUs for each CJ. Preserve
native and portable results as separate rows from the same frozen candidate.

Focused tests exercise CPU and GPU requests through native resource/identity
mapping, queued/running cancel,
late completion, unknown/duplicate/step accounting, missing manifests,
retention through receipt, active result reads during shutdown and a real
NPTrainer worker with a mocked scheduler. These tests submit no Slurm jobs.
Integration tests also use the normal client JSON configurator, canonical site
policy and TaskWorkerExecutor, including an explicit zero-GPU override of a
GPU site default and replay without a second physical launch.
GPU visibility and scheduler responses are simulated locally; actual GPU
computation, allocation release and queued-capacity reuse require the deployed
GPU cases below. CPU successes alone do not qualify GPU execution.

Bounded single-node qualification fixture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The qualification exporter explicitly requests one CPU and 2048 MiB for the
server job and each resident CJ or task worker, with a 20-minute limit. The
server always requests zero GPUs. A previous fixture omitted the server
memory request; a scheduler default reserved all node memory and prevented
the clients from starting. A different pair of parent allocations expired
at their 30-minute limit. These failures must remain in the campaign record;
a later GPU success does not replace a missing CPU resident control.

``tests/integration_test/data/slurm_task_execution/server.yaml`` and
``client.yaml`` are supported ``nvflare deploy prepare`` inputs for this
single-node fixture. Replace their example Python/environment paths with
the exact candidate environment already on the authorized host. The client
parent has a one-day limit, while its job launcher gives CPU CJs a one-hour
limit. Use a distinct client internal port (18111 and 18112), and server
port 18110, after checking the existing owned services. The fixture keeps
mTLS and uses loopback only because all allocations share one host. Server
parent startup remains a stable service, as required by the deployment tool.

Prepare fresh output directories from the existing on-host startup kits,
using the normal deployment command. Install the exact trusted task launcher
pair described above in each client's site resources. Set ``cj_cpu_cores=1``
and ``cj_memory_mb=1024``; task defaults are one CPU, 2048 MiB and zero GPUs.
The exported metadata selects CPU or GPU compute for each row. Keep the
original startup kits and prior prepared directories private and intact.

Parent CPU/memory/GPU requests use native ``sbatch`` options; ``mem`` and
``cpus-per-task`` are not supported keys in ``parent.sbatch_directives``.
Submit each new client parent once, recording its returned allocation ID::

    sbatch --parsable --no-requeue --nodes=1 --ntasks=1 --cpus-per-task=1 \
      --mem=2048M --gres=none --time=1-00:00:00 --export=NIL \
      --job-name=nvflare-site-1-parent \
      --output=prepared/site-1/parent-slurm-%j.out \
      prepared/site-1/startup/parent.slurm

Repeat for site-2 with its own output and script. Do not submit replacements
while previous parents still run. The coordinator must recheck lease
ownership, scheduler availability and existing workloads before changing an
owned deployment. Changing YAML does not change an already-running
allocation's resources or time limit.

At the peak of a task-mode row the two parents request 4 GiB, the server job
2 GiB, two CPU CJs 2 GiB and two workers 4 GiB: 12 GiB in total, plus headroom
for the service parent and host. A job-mode row requests 10 GiB in total.
Verify available CPU slots, actual memory TRES and two free GPUs as well;
the scheduler can round CPU allocations for the host's topology. Run the
four rows serially so resident and task GPU requests never overlap.

The portable task-mode profile requests 15 GiB in total, and its job-mode
profile requests 12 GiB, plus service/host headroom. Check CPU capacity too;
the portable task CJs and workers each request two CPUs. Keep the native
profile's resource record separate rather than applying its 12/10 GiB budget
to these larger portable cases.

Record each row separately as requested, running, blocked, failed or
verified-passed. Required rows are CPU/job, CPU/task, CUDA/job and CUDA/task,
all from the same frozen candidate and configuration. All four must pass
with one server, two sites and both compute rounds. Capture participant
imports, source/config hashes, job and native allocation identities, numeric
compute checksums, returned metrics and the persisted server model. CUDA
rows must demonstrate actual allocation, kernels and synchronization on each
site. A row is not verified-passed until supported completion/shutdown and
accounting show no CJ/task allocation, worker process, GPU compute process
or task reservation remains; retained CPU parent services are expected and
must be identified separately. Preserve earlier failures instead of replacing
their statuses. This fixture does not itself submit any product job or
qualify cancellation, cross-host storage or crash recovery.

Submitted metadata and live resource evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exported ``meta.json`` is not proof of submitted resource requirements.
Run the real ``nvflare job submit`` path, then retrieve the server-stored job
metadata and retain it with the job ID and source/config hashes. Compare its
compute and CJ budgets with the intended values before accepting a run.
Literal ``@default`` keys, and the ``@ALL`` CJ marker for an all-client task
deployment, must survive CLI staging and server validation. A missing or
renamed key is a failed qualification even if computation completes.

For each owned CJ and task allocation, retain its submitted scheduler request
and live Slurm requested/allocated TRES, including CPU, memory and GPU counts.
Join those records to the job/site/attempt identity and terminal accounting.
The task-mode CJ must match only its coordinator budget, while every task
must match the compute budget. In job mode compute belongs to the resident
CJ. Explain scheduler CPU rounding against the explicit request; an inherited
default or ambient GPU visibility cannot replace allocation evidence.

The local qualification tests pass exported native and portable jobs through
the same ``prepare_job_config`` staging call used by the submit CLI, then
resolve their compute and CJ budgets into real Slurm submission arguments.
This detects metadata conversion regressions locally. It does not replace
the required server-stored metadata and live allocation checks above.

Additional deployed qualification requires an authorized test scheduler/account,
the shared filesystem and matching software environment, CPU CP/CJ
allocations, the completed shared runtime integration, and an available GPU
partition for GPU compute and resource-reuse cases. With uniquely owned test allocations,
record allocation/step identities, scheduler release, retained output and
server receipts. Queue a second task for the constrained GPU capacity and
prove it runs while the first CJ is still uploading. Exercise queued/running
cancellation, leader exit with remaining descendants, compute failure and
accounting outages. Repeat the required POC/secure production and resident
compatibility cases. Mock results and local CPU workers do not qualify any
of these deployed criteria. No shared scheduler submissions or configuration
changes are part of the local reproduction commands.
