.. _slurm_job_launcher:

##################
Slurm Job Launcher
##################

In this guide, the *parent* is the long-running NVFlare client parent process
(CP) or server parent process (SP). It launches a separate client job process
(CJ) or server job process (SJ) for each federated job. The Slurm job launcher
submits each CJ or SJ as a Slurm batch job. Slurm selects nodes and enforces the
allocation; NVFlare submits, monitors, and cancels jobs owned by the current
parent.

The *prepare host* runs ``nvflare deploy prepare``. The optional *submission
host* runs ``sbatch`` to place a client parent in Slurm. The *runtime parent
host* runs the CP or SP, and the *compute nodes* run its CJ or SJ allocations.
A login or service host is a runtime parent host outside a Slurm allocation.

Choose an execution backend with ``job_launcher.sandbox``:

.. list-table::
   :header-rows: 1

   * - Value
     - Execution
     - Use
   * - ``apptainer``
     - Launcher-managed Apptainer container
     - Isolation after validation on the target cluster
   * - ``pyxis``
     - Read-only Pyxis/Enroot container
     - Trusted container packaging
   * - ``none``
     - Python directly in the allocation
     - Site-trusted code; required for application-owned multi-node fan-out

Prerequisites
=============

Before starting a parent, verify:

- Slurm 23.02 or later provides working ``sbatch``, ``squeue``, ``sacct``, and
  ``scancel`` commands on the runtime parent host. Parent bootstrap resolves
  these commands and verifies the version; 23.02 is the minimum because the
  launcher uses ``sbatch --export=NIL``. Production sites should run a Slurm
  release that is still supported by SchedMD.
- ``slurmdbd`` accounting is enabled and ``sacct`` responds. The default
  ``AccountingStoreFlags`` is sufficient.
- The cluster is not federated, and submission plugins do not redirect jobs to
  another cluster.
- The parent uses a dedicated site account with a consistent numeric UID and
  compatible group access on the parent host, shared filesystem, and compute
  nodes.
- The runtime workspace, images, datasets, and secret-mount sources are visible
  at the same absolute paths on all participating nodes. The shared filesystem
  supports ``O_EXCL`` and atomic rename.
- Compute nodes can reach ``parent_host:internal_port``, or the site uses the
  shared-file worker channel instead (see :ref:`slurm_shared_file_channel`).
  Multi-node jobs also require connectivity among their allocated nodes.
- Slurm partition, association, QOS, reservation, cgroup, and device policies
  enforce the site's resource limits.

The prepare output is the runtime workspace. The parent and worker allocations
must see it at the same absolute path. Every workspace must be private, owned
by the runtime account, and dedicated to one NVFlare site or federation.

For Apptainer, enable unprivileged user namespaces and install Apptainer on all
eligible nodes. Validate its filesystem, process, cgroup, and GPU isolation on
the production cluster. For Pyxis, install and configure Pyxis/Enroot on all
eligible nodes and ensure ``srun`` is available after ``setup``. Launcher-owned
multi-node fan-out (``node_command``) also requires ``srun`` after ``setup``.

The environment selected by ``python_path`` must contain a compatible NVFlare
installation and the job dependencies. This applies to the host environment in
bare mode and to the image for Apptainer and Pyxis. A ``PYTHONPATH`` override
used only to start the parent does not install NVFlare in the worker environment.

The prepare, submission, and runtime parent hosts may differ. The prepare host
does not need Slurm commands. The submission host that runs the generated
``submit_command`` needs ``sbatch`` on ``PATH``. The runtime parent host needs
all four parent commands after its service environment or
``parent.environment_setup`` has run.

Configure the Site
==================

Create ``slurm.yaml`` beside the startup kit. This example runs the parent on a
login or service host:

.. code-block:: yaml

   runtime: slurm
   job_launcher:
     sandbox: apptainer
     image: /lustre/images/nvflare-prod.sif
     python_path: /usr/bin/python3
     parent_host: nvflare-site1.internal
     sbatch_directives:
       partition: fl-gpu
       account: proj123
       time: "12:00:00"
     setup: |
       source /etc/profile.d/modules.sh
       module load apptainer
     pending_timeout: 600

Important keys are:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``sandbox``
     - Required: ``apptainer``, ``pyxis``, or ``none``.
   * - ``image``
     - Existing absolute image file required by Apptainer and Pyxis. Omit in
       bare mode.
   * - ``python_path``
     - Required absolute worker interpreter path in the selected execution
       environment, with a compatible NVFlare installation.
   * - ``parent_host``
     - Compute-reachable parent host. Required when the parent is not in a Slurm
       allocation.
   * - ``sbatch_directives``
     - Site defaults. Supported keys are ``partition``, ``account``, ``qos``,
       ``time``, ``constraint``, and ``reservation``.
   * - ``setup``
     - Trusted host-side Bash run in the batch job before the worker starts.
   * - ``forward_env``
     - Environment names whose post-``setup`` values should reach the worker.
   * - ``executables``
     - Optional explicit paths for Slurm and backend commands. Parent-side
       paths are preserved in the workspace, then resolved and validated on the
       runtime parent host.
   * - ``internal_port``
     - Worker-to-parent port; default ``8102``.
   * - ``poll_interval``
     - Scheduler polling interval; default ``10`` seconds.
   * - ``pending_timeout``
     - Time limit starting at the first observed pending state; default ``600``
       seconds. A job may lower it.

``setup`` starts with the minimal environment from ``sbatch --export=NIL``.
Source module initialization explicitly. Put fixed values in study ``env``;
use ``forward_env`` only for values created by setup.

Prepare and Start
=================

Prepare directly into the shared runtime workspace:

.. code-block:: shell

   nvflare deploy prepare ./site-1 \
       --config ./slurm.yaml \
       --output /lustre/proj123/nvflare/site-1

Run one parent per workspace. Re-running prepare with the same output replaces
the complete workspace. Stop the parent and preserve required runs, snapshots,
and server job storage before updating it.

Start a parent on a login or service host:

.. code-block:: shell

   /lustre/proj123/nvflare/site-1/startup/start_slurm.sh

To run a client parent in a Slurm allocation, add a client-only ``parent``
block and omit ``job_launcher.parent_host``:

.. code-block:: yaml

   parent:
     sbatch_directives:
       partition: batch
       account: proj123
       time: "7-00:00:00"
     environment_setup: |
       source /lustre/proj123/venv/bin/activate

Then run the ``submit_command`` printed by prepare on a submission host where
``sbatch`` is on ``PATH``. It has this form:

.. code-block:: shell

   sbatch --parsable \
       --output=/lustre/proj123/nvflare/site-1/parent-slurm-%j.out \
       /lustre/proj123/nvflare/site-1/startup/parent.slurm

The parent script runs ``parent.environment_setup`` before starting NVFlare.
Parent bootstrap then resolves ``sbatch``, ``squeue``, ``sacct``, and
``scancel`` once and keeps their canonical paths in memory for that process.
An explicitly configured path may therefore point to a cluster-managed stable
symlink such as ``.../slurm/current/bin/sbatch``; a restarted parent resolves a
new target after a cluster upgrade without re-preparing the workspace.

An explicit ``parent_host`` always wins. Otherwise an allocated parent uses
``SLURMD_NODENAME``. A parent outside an allocation without ``parent_host``
cannot launch jobs. NVFlare does not guess or resolve a host name.

Server kits reject ``parent``. Run the server parent on a stable host with a
stable external NVFlare federation endpoint.

.. _slurm_shared_file_channel:

Shared-File Worker Channel
==========================

When compute nodes cannot open a TCP connection to the parent host but share a
POSIX-coherent filesystem such as Lustre with it, the worker-to-parent channel
can run over shared files instead of TCP. Configure the client kit's
``local/comm_config.json`` before running prepare:

.. code-block:: json

   {
     "backbone": {"connect_generation": 1},
     "internal": {
       "scheme": "shared-file",
       "resources": {
         "root_dir": "/lustre/proj123/nvflare/site-1-cellnet",
         "connection_security": "clear"
       }
     }
   }

``root_dir`` must be an absolute path visible at the same path on the parent
host and all compute nodes. ``connect_generation: 1`` routes all job traffic
through the parent, so workers need no network connectivity at all.
``nvflare deploy prepare`` preserves a file-based comm config as-is and does
not apply the TCP host and port patch; ``internal_port`` and ``parent_host``
are then not used for the worker channel.

At runtime the parent creates a listener directory under ``root_dir`` and
passes its ``shared-file://0/...`` URL to each worker unchanged. Apptainer and Pyxis
jobs bind-mount the listener directory read-write at the same path inside the
container automatically; bare jobs use it directly. Directories are created
with mode ``0o770`` and log files with ``0o660`` regardless of umask.
Directory permissions are the only access control on this channel, so keep
``root_dir`` owned by the dedicated site account with no wider group access
than required.

Polling intervals, lease timing, and fsync behavior are tunable through the
``internal.resources`` map; see the ``FileDriver`` documentation in
``nvflare.fuel.f3.drivers.file_driver`` for the parameters and their
filesystem metadata cost. At the defaults an idle connection issues roughly
1.4 client-side metadata syscalls per second; raising ``max_poll_interval``
reduces idle load proportionally at the cost of first-message latency, and
data transfers are unaffected by the poll settings.

Study Settings
==============

Site-owned study settings belong in ``local/study_runtime.yaml`` and are
re-read for every launch:

.. code-block:: yaml

   format_version: 2
   studies:
     pathology:
       container:
         image: /lustre/images/pathology.sif
       datasets:
         slides:
           source: /lustre/data/pathology/slides
           mode: ro
       env:
         MODEL_FAMILY: vit-large
       secret_env:
         DB_PASSWORD:
           source: PATHOLOGY_DB_PASSWORD
       slurm:
         partition: fl-gpu-large
         account: pathology-project

Each dataset is mounted at ``/data/<study>/<dataset>`` inside the container;
the example above mounts ``slides`` at ``/data/pathology/slides``.

A study can override the site sandbox, setup, partition, account, and QOS. It
can also provide an image, environment, dataset mounts, secret environment, and
read-only secret mounts. A Slurm ``secret_env`` source names a variable in the
parent environment; its value is passed through a temporary private file and is
not written to the batch script or scheduler command.

Mount sources must be absolute paths outside the runtime workspace and must
exist on the compute nodes that can run the job. Bare mode
rejects container, dataset, and secret-mount settings because it has no mount
namespace. Migrate legacy
``local/study_data.yaml`` files to ``study_runtime.yaml`` before using Slurm.

Job Settings
============

Jobs put portable GPU totals in ``resource_spec`` and Slurm-only settings in
``launcher_spec``. This is a valid single-node container request:

.. code-block:: json

   {
     "resource_spec": {
       "site-1": {"num_of_gpus": 1}
     },
     "launcher_spec": {
       "site-1": {
         "slurm": {
           "image": "/shared/images/nvflare-job.sif",
           "cpus_per_node": 8,
           "mem_per_node": 32768,
           "time": "02:00:00",
           "pending_timeout": 300
         }
       }
     }
   }

Supported job keys are ``image``, ``nodes``, ``gpus_per_node``,
``cpus_per_node``, ``mem_per_node`` (MiB), ``time``, ``pending_timeout``, and
``node_command``. A job image requires normal BYOC authorization and takes
precedence over the study and site images. Job and study images are rejected on
any site whose effective sandbox is ``none``.

A positive multi-node ``num_of_gpus`` requires ``gpus_per_node``;
whenever both are supplied, ``num_of_gpus`` must equal
``nodes * gpus_per_node``.

A multi-node client job that also sets ``node_command`` requests a
launcher-owned node group: the launcher starts one task per allocated node,
runs the normal client job process on node rank 0, and runs ``node_command``
— the worker command for the non-zero node ranks — on every other node with
the node-group environment (``NVFL_NNODES``, ``NVFL_NODE_RANK``,
``NVFL_MASTER_ADDR``, ``NVFL_MASTER_PORT``) exported to all tasks. Node
groups work under every sandbox: with ``apptainer`` or ``pyxis``, all user
code on every node runs inside the configured container, exactly as in a
single-node container job. ``node_command`` executes in the deployed job app
directory as the submitting user, with the same trust as the job's own
training code; secret references are not supported in it.

Jobs built with the FedJob/Recipe API do not write ``node_command`` by hand:
job export fills it from the site's external training command
(``ScriptRunner``'s resolved ``command``) whenever a launcher block requests
``nodes > 1``, and enforces ``launch_once=True`` (the ScriptRunner default).
An explicit ``node_command`` always wins; for jobs that set it directly, the
platform does not verify that it matches the rank-0 training command. For
PyTorch jobs,
``python3 -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=<G> -- <script> <args>``
maps this environment onto torchrun rendezvous arguments and is intended to be
both the job's training command and its ``node_command``:

.. code-block:: json

   {
     "resource_spec": {
       "site-1": {"num_of_gpus": 16}
     },
     "launcher_spec": {
       "site-1": {
         "slurm": {
           "nodes": 2,
           "gpus_per_node": 8,
           "node_command": "python3 -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 -- custom/client.py"
         }
       }
     }
   }

Without ``node_command``, a multi-node allocation keeps the client job process
alone on the first node and the application owns any fan-out; this mode
requires effective ``sandbox: none`` because only a bare client job process
can reach ``srun``.

Security and Operations
=======================

The worker-to-parent internal channel is clear TCP, matching the Kubernetes
launcher, or clear shared-file I/O when the file transport is configured (see
:ref:`slurm_shared_file_channel`). Use it only on a trusted or isolated site
network or filesystem. This does not change the configured security of the
external NVFlare federation channel.

Working accounting is mandatory. The parent refuses to start if ``sacct`` is
unavailable. A later scheduler or accounting outage leaves affected jobs
non-terminal and retries; it never assumes that a missing observation means a
job has stopped.

NVFlare stores transient launch artifacts under ``<prepare-output>/.nvflare_slurm``.
The live parent removes a job's artifacts after launch failure or terminal
completion. User abort and pending timeout verify ownership before ``scancel``;
normal framework shutdown terminates running handles through the same path
before the Slurm launcher closes launch admission.

Slurm job names include the first 32 characters of the NVFlare site name and a
short job hash, so operators can distinguish sites sharing one Slurm user. Job output is
``<run-dir>/slurm-<slurm-job-id>.out``. Use ``squeue`` for live state and
``sacct`` for completed jobs. Preserve the complete workspace and relevant
scheduler records for investigation.

After a parent crash, the launcher does not cancel surviving allocations or
remove their job artifacts; this matches the Docker and Kubernetes launchers.
A leftover job directory blocks relaunch of that job ID until an operator
verifies that no old allocation uses it and removes the directory.

An ``sbatch`` timeout fails the FL dispatch even if Slurm accepted the job.
Artifact removal prevents a pending allocation from starting unless the same
job ID is relaunched before it starts.

To change the workspace, stop the parent and prepare to a new ``--output``.
Preparing to an existing output replaces it, so copy any data that must survive
before running prepare.

After a server job starts, clients wait for its SJ to become available. The
SJ's Slurm queue and startup time must fit within the client-side
``max_runner_sync_timeout`` (60 seconds by default). Configure this value in
the job's client ``config_fed_client.json`` when the site needs a longer bound;
see :ref:`timeout_troubleshooting`.

Before production use, test on the target cluster:

#. Successful, failed, timed-out, pending-aborted, and running-aborted jobs.
#. Parent restart with a live job and workspace replacement behavior.
#. Compute-node connectivity to the parent and multi-node collectives when used.
#. Slurm accounting, association, QOS, partition, cgroup, and GPU enforcement.
#. The selected backend's filesystem view, environment, secret handling, exit
   status, and CPU/GPU device visibility.

Do not describe Apptainer as a production sandbox until those isolation checks
pass on the deployment cluster.
