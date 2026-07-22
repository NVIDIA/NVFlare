# Slurm Job Launcher Design

Status: implemented

This document describes the architecture and behavioral contract of the NVFlare Slurm job launcher. Operator setup
and configuration are in `docs/user_guide/admin_guide/deployment/slurm_job_launcher.rst`.

## Purpose and scope

The launcher runs each NVFlare client job process (CJ) or server job process (SJ) as a Slurm batch job. Slurm owns
placement and resource enforcement. NVFlare submits, monitors, and cancels jobs started by the current parent process
through the existing `JobLauncherSpec` and `JobHandleSpec` interfaces.

Supported execution modes are:

- one-node bare, Apptainer, and Pyxis jobs;
- multi-node client jobs as launcher-owned node groups, bare or containerized;
- trusted bare multi-node jobs with application-owned fan-out.

The deployment targets one Slurm cluster, with scheduler routing controlled by trusted site policy. The internal
worker-to-parent connection is clear TCP and runs on a trusted site network.

## Main components

```text
NVFlare parent
    |
    | JobLauncherSpec / JobHandleSpec
    v
SlurmJobLauncher -> SlurmJobManager -> sbatch, squeue, sacct, scancel
                         |
                         v
                 Slurm CJ or SJ allocation
```

The launcher resolves site, study, and job configuration into one immutable launch plan. The manager owns scheduler
access and transient job artifacts. A live `SlurmJobHandle` holds the scheduler ID, whether cancellation was
requested, whether it was a user abort, the pending timer, accounting miss count, and final result. After launch,
the client or server framework calls
`JobHandle.wait()` once in a background thread. For Slurm, `wait()` polls the scheduler and accounting until the
allocation is terminal, cleans the job artifacts, and returns control to normal NVFlare completion handling.

## Deployment and workspace

`nvflare deploy prepare --output <workspace>` creates the complete runtime workspace directly. The output must be a
stable shared-filesystem path visible to the parent and eligible compute nodes.

The generated `start_slurm.sh` is an optional convenience wrapper. It sets the runtime workspace and starts the
parent in the foreground. An external service manager may perform the same operations.

Prepare can also create an optional `parent.slurm` script for running a client parent in a Slurm allocation. Server
parents run on a service or login host with a stable public endpoint.

The prepare host does not resolve scheduler commands. It preserves optional configured absolute paths in the workspace,
and generated parent-submission guidance uses `sbatch` from the submission host's `PATH`. After the parent reaches
its actual runtime host and trusted environment setup has run, bootstrap resolves `sbatch`, `squeue`, `sacct`, and
`scancel`, verifies Slurm 23.02 or later, and freezes canonical paths in memory for that parent process. A restart
therefore follows changes to a cluster-managed stable symlink without requiring a new prepare.

Operators use a separate output for each NVFlare site or federation and run one parent per workspace. Preparing
again with the same output replaces the complete workspace, including runtime data. Stop the parent and preserve
required runs, snapshots, and server job data before replacement.

The workspace and all configured images or mounts must be visible at the same absolute paths from the submit host
and eligible compute nodes. The filesystem must support coherent exclusive create and atomic rename.

## Configuration ownership

| Owner | Source | Controls |
| --- | --- | --- |
| Site | `slurm.yaml` and prepare `--output` | workspace, default sandbox/image, scheduler policy, setup, commands, timeouts |
| Site study policy | `local/study_runtime.yaml` | study image, mounts, environment, sandbox/setup/routing overrides |
| Job | `launcher_spec` and `resource_spec` | BYOC-authorized image, topology, node command, CPU, memory, time, GPU total |

The effective image is job, then study, then site. A job image uses the same BYOC authorization as Docker and
Kubernetes. Container images must be absolute files visible on the compute nodes. Partition, account, QOS, sandbox,
setup commands, and scheduler options remain under site or study policy.

The portable `resource_spec[site].num_of_gpus` remains the total GPU request. Slurm topology is supplied through the
effective Slurm launcher block. A job may reduce, but not increase, the site's pending timeout.

Site files are trusted policy. They are checked for structure and for mistakes that could expose launcher-owned
workspace or credential paths, but are not treated as hostile input. Job values and scheduler output remain
untrusted and are validated at their boundaries.

## Job artifacts and scheduler identity

Each launched job uses one transient directory:

```text
.nvflare_slurm/jobs/<sha256-job-id>/
  batch.sh
  node.sh        # multi-node node groups only
  secret.env
  sandbox_root/  # container modes only
```

The job key is the SHA-256 digest of the NVFlare job ID. Runtime identity is deterministic:

```text
job name = nvfl-<first-32-site-name-characters>-<first-8-job-key>
comment  = nvfl:<job-id>
```

The scheduler ID exists only in the live handle. The launcher removes job artifacts after submission failure or
terminal completion. A leftover directory blocks relaunch of the same job ID until an operator verifies that no old
allocation uses it and removes it.

Duplicate protection is also in memory: one parent refuses a second launch while its live-handle map contains the
same job ID.

## Submission

Launch proceeds as follows:

1. Resolve one launch plan and reject a duplicate live handle for the same job.
2. Create the job artifact directory and render its batch and secret-environment files.
3. Invoke `sbatch --parsable` once with structured arguments and a scrubbed scheduler environment.
4. Accept exactly one parsed bare job ID, add its handle to the live-handle map, and return it.

An invocation timeout, exception, or output other than one line matching the Slurm job-ID format fails the launch.
The manager immediately removes the job's artifacts.

An out-of-contract `job-id;cluster` result triggers one best-effort `scancel -M`, a critical log, and an
infrastructure launch failure.

## Monitoring and results

Live lookup uses `squeue --name=<job-name>` for the submitting user and selects the row with the handle's job ID.
The site-qualified name avoids collisions when several NVFlare sites share a Slurm user and run the same federated
job. Rows with other IDs are still ignored, and the selected row must match the numeric UID, full name, and derived
comment.

When a job is absent from `squeue`, `sacct --jobs=<job-id>` is authoritative. The returned allocation must match the
exact ID, deterministic job name, and submitting user.

An accounting outage leaves the handle non-terminal. If five successful `sacct` queries at least six seconds apart
all return no record for a known job, the launcher reports an infrastructure exception. It does not infer a worker
result from the missing record.

`pending_timeout` starts when the scheduler first reports `PENDING`, `CONFIGURING`, `REQUEUE_HOLD`,
`RESV_DEL_HOLD`, or `SPECIAL_EXIT`. Other ordinary live states remain active. Submission disables requeue, and the
batch script refuses a restarted allocation before starting the worker.

Slurm `State` and `ExitCode` determine the scheduler fallback result. The generic NVFlare
`_process_rc.txt` handling may refine the application result after `wait()`.

## Cancellation, startup, and shutdown

`terminate()` records user-abort intent and requests cancellation. A pending-timeout expiry also requests
cancellation. Every cancellation repeats live ownership verification before calling `scancel -Q --me`. A
current-parent user abort maps scheduler `CANCELLED` or `COMPLETED` to `ABORTED`, so abort intent wins a race with
normal completion.

At startup, the manager validates the private runtime workspace and briefly retries the required accounting probe.

During normal shutdown, the framework starts running-job termination before firing `SYSTEM_END`; captured handles
are terminated through that framework abort path. The Slurm `SYSTEM_END` handler only closes launch admission after
any in-progress submission boundary; it does not perform a second cancellation sweep.

## Accepted crash limitations

- Running a second parent on an already-active workspace is unsupported. Both parents would share deterministic job
  artifact paths and scheduler ownership markers. The launcher provides no refusal mechanism.
- After a parent crash, surviving Slurm allocations and their job artifacts are not cleaned up by the launcher. This
  matches the Docker and Kubernetes launchers; allocations remain bounded by their Slurm wall-time and normal
  FL-layer rejection.
- An `sbatch` timeout fails the FL dispatch even if Slurm accepted the job, matching a timed-out Docker or Kubernetes
  create. Removing `secret.env` prevents a pending allocation from starting unless a later launch of the same
  job ID recreates that deterministic path first.

## Execution backends and secrets

| Sandbox | Execution | Trust model | Multi-node |
| --- | --- | --- | --- |
| `none` | worker directly in the allocation | trusted site workload | yes; launcher-owned node group or application-owned fan-out |
| `apptainer` | contained unprivileged `apptainer exec` | cluster-accepted isolation | yes; launcher-owned node group only |
| `pyxis` | read-only Pyxis/Enroot `srun` step | trusted container packaging | yes; launcher-owned node group only |

Every backend uses `--export=NIL`, refuses requeue, and passes the standard NVFlare worker arguments without shell
re-parsing. Bootstrap credentials and study `secret_env` values are written to a mode-0600 file, sourced with tracing
disabled, and deleted before the worker starts. Bootstrap credentials are delivered to the job process through the
shared `JobProcessEnv` contract and do not appear in its command line.

Container mounts have normalized absolute destinations, protected launcher paths cannot be shadowed, and secret
mounts are read-only. Apptainer uses the accepted containment and mount restrictions; Pyxis uses a read-only image,
no home mount, and no image entrypoint. Backend isolation still depends on cluster configuration and must be
accepted by the site.

## Multi-node node groups

A client job opts into a launcher-owned node group by combining `nodes > 1` with a `node_command` in its effective
Slurm launcher block. The single allocation starts one task per node: node rank 0 runs the normal CJ worker
unchanged, every other node runs the job's `node_command`. Extra nodes never register with the server and carry no
FL identity; cross-node coordination (rendezvous, collectives) belongs to the training framework. Without
`node_command`, a multi-node allocation keeps the previous behavior: the CJ runs alone on the first node and the
application owns any fan-out.

### Environment contract

The batch script, which always executes on the first node of the allocation, exports a scheduler-neutral contract
and delegates to one `srun --nodes=N --ntasks=N --ntasks-per-node=1` invocation of the generated `node.sh`:

| Variable | Value |
| --- | --- |
| `NVFL_NNODES` | `SLURM_JOB_NUM_NODES` |
| `NVFL_NODE_RANK` | `SLURM_NODEID`, exported per task by `node.sh` |
| `NVFL_MASTER_ADDR` | `SLURMD_NODENAME` of the batch node, which is node rank 0 |
| `NVFL_MASTER_PORT` | `29400 + SLURM_JOB_ID % 1000`, deterministic per allocation and collision-safe on shared nodes |

`node.sh` dispatches on the rank: rank 0 executes the standard worker command, so the CJ connects to the parent
and reports the job result exactly as a single-node job; every other rank executes the `node_command` argv in the
deployed job app directory. Both paths inherit the batch environment (sourced secrets, `PYTHONPATH`, study env,
forwarded names) because the script exports `SLURM_EXPORT_ENV=ALL` before the fan-out. The variable names carry
no scheduler meaning, so the same `node_command` can run under any launcher that adopts the contract.

`node_command` is job-owned and validated at the launch boundary: a single-line, shell-lexable, non-empty string,
split once into argv and rendered fully quoted, never re-parsed by a shell. It executes as the submitting user
under the effective sandbox, with exactly the trust of the BYOC training code the rank-0 CJ launches itself. It
is rejected for server jobs, for `nodes: 1`, and when the deployed job app directory is missing.

### Sandboxed node groups

Node groups compose with every sandbox because the ordering is fixed: scheduler fan-out first, on the bare
allocation, containers second, as per-node leaves. All user code, on every rank, runs under the effective sandbox
with the launcher-standard isolation flags. Application-owned fan-out (multi-node without `node_command`) still
requires effective sandbox `none`, since only a bare CJ can reach `srun`.

| Sandbox | Fan-out | Containerization |
| --- | --- | --- |
| `none` | bare `srun` of `node.sh` | none |
| `apptainer` | bare `srun` of `node.sh` on each node | `node.sh` starts one `apptainer exec` per rank, `--pwd` run dir (rank 0) or app dir (others) |
| `pyxis` | one `srun` carrying the usual container flags | Pyxis creates the per-task container; `node.sh` dispatches inside it |

Environment delivery follows each backend's existing mechanism: bare tasks inherit the exported batch environment;
Apptainer receives the contract through `APPTAINERENV_*` mirrors (`NVFL_NODE_RANK` per task inside `node.sh`);
Pyxis adds the contract names to the shared `--export`/`--container-env` list, with `SLURM_NODEID` provided per
task by Slurm. For Pyxis the job artifact directory is bind-mounted read-only because `srun` starts `node.sh`
inside the container; the secret file is already deleted before any task starts, and the image must provide
`bash`.

### Lifecycle and results

The fan-out uses `--kill-on-bad-exit=0`: a failing worker node must not kill rank 0 before the CJ reports the
training failure through the normal task path; surviving ranks observe the loss through the training framework's
own failure detection. Rank 0 remains authoritative for the application result via `_process_rc.txt`; the
scheduler fallback still uses allocation `State` and `ExitCode`, which reflect the highest task exit code.
Cancellation, monitoring, and pending-timeout handling are unchanged because the node group is one allocation
with one scheduler identity. Between training rounds the non-zero ranks hold their nodes idle, which is inherent
to a static allocation.

### Framework helpers

The contract is the minimal "single-coordinator rendezvous" set that PyTorch, DeepSpeed, XGBoost trackers, Ray,
and JAX all self-assemble from; `node_command` may be any executable, including a plain shell wrapper reading the
variables itself. `nvflare.app_common.multinode` provides the shared parsing (`NodeGroup.from_env`, the `--`
command boundary); a framework helper is a thin translation of a `NodeGroup` into framework arguments. The first
consumer, `nvflare.app_opt.pt.torchrun_node`, maps the contract onto torchrun c10d rendezvous arguments (with a
configurable join timeout for the window before the CJ starts training) and degrades to standalone single-node
torchrun when the contract is absent, so the same command line serves as the job's rank-0 training command and as
its `node_command`:

```text
python3 -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 -- custom/client.py --epochs 2
```

Known contract limits: frameworks that need every member's address up front (for example `TF_CONFIG`) would need
an additive node-list variable, and PMI-launched MPI does not fit the per-node-exec model because PMI expects the
scheduler to start the ranks themselves.

## Required environment

The runtime parent requires Slurm 23.02 or later because the launcher uses `sbatch --export=NIL`. Parent bootstrap
requires working `sbatch`, `squeue`, `sacct`, and `scancel` commands and working `slurmdbd` accounting. Default
`AccountingStoreFlags` is sufficient. It targets a single, non-federated cluster, and site plugins must preserve
local submission routing. Apptainer or Pyxis/Enroot must be installed on eligible nodes when selected; Pyxis and
launcher-owned multi-node fan-out also require `srun`. Production sites should use a Slurm release that is still
supported by SchedMD.

The environment selected by `python_path` must contain a compatible NVFlare installation. The launcher sets the
worker `PYTHONPATH` to the resolved job and site custom directories, so a source overlay used only by the parent is
not a worker installation.

The parent address comes from explicit `parent_host`, which always wins, or `SLURMD_NODENAME` when the parent itself
runs inside a Slurm allocation. A parent outside an allocation therefore requires `parent_host`.
