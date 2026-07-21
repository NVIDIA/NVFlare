.. _slurm_job_launcher:

##################
Slurm Job Launcher
##################

The Slurm job launcher submits every NVFlare client job process (CJ) and server
job process (SJ) as a Slurm batch job. Slurm selects nodes and enforces the
allocation; NVFlare submits, monitors, and cancels jobs owned by the current
parent process.

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
     - Site-trusted code; required for multi-node jobs

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
- Compute nodes can reach ``parent_host:internal_port``. Multi-node jobs also
  require connectivity among their allocated nodes.
- Slurm partition, association, QOS, reservation, cgroup, and device policies
  enforce the site's resource limits.

Worker allocations use the staged workspace and do not need the prepared output.
When ``parent.slurm`` starts a client parent in an allocation, that allocation
must also see the prepared output at its configured absolute path. Every
workspace must be private and owned by the runtime account. One without a Slurm
runtime identity must also be empty; a previously staged workspace may contain
runtime data but must have a valid identity for the same site.

For Apptainer, enable unprivileged user namespaces and install Apptainer on all
eligible nodes. Validate its filesystem, process, cgroup, and GPU isolation on
the production cluster. For Pyxis, install and configure Pyxis/Enroot on all
eligible nodes and ensure ``srun`` is available after ``setup``.

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
   workspace_path: /lustre/proj123/nvflare/site-1

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
   * - ``workspace_path``
     - Required stable runtime root, outside the provisioned and prepared kits.
       One site deployment owns the path.
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
       paths are preserved in the kit, then resolved and validated on the
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

Prepare, Stage, and Start
=========================

Prepare and stage the kit:

.. code-block:: shell

   nvflare deploy prepare ./site-1 \
       --config ./slurm.yaml \
       --output /opt/nvflare/prepared/site-1-slurm

   nvflare deploy slurm stage /opt/nvflare/prepared/site-1-slurm

Staging creates or verifies the runtime workspace identity and installs the
prepared kit. Run one parent per workspace and stop it before staging an update.
Re-run prepare and stage to update kit content; the runtime identity, runs,
snapshots, and server job storage remain in place.

Start a parent on a login or service host:

.. code-block:: shell

   /opt/nvflare/prepared/site-1-slurm/startup/start_slurm.sh

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

Then stage normally and run the ``submit_command`` printed by prepare on a
submission host where ``sbatch`` is on ``PATH``. It has this form:

.. code-block:: shell

   sbatch --parsable \
       --output=/opt/nvflare/prepared/site-1-slurm/parent-slurm-%j.out \
       /opt/nvflare/prepared/site-1-slurm/startup/parent.slurm

The parent script runs ``parent.environment_setup`` before starting NVFlare.
Parent bootstrap then resolves ``sbatch``, ``squeue``, ``sacct``, and
``scancel`` once and keeps their canonical paths in memory for that process.
An explicitly configured path may therefore point to a cluster-managed stable
symlink such as ``.../slurm/current/bin/sbatch``; a restarted parent resolves a
new target after a cluster upgrade without re-preparing the kit.

An explicit ``parent_host`` always wins. Otherwise an allocated parent uses
``SLURMD_NODENAME``. A parent outside an allocation without ``parent_host``
cannot launch jobs. NVFlare does not guess or resolve a host name.

Server kits reject ``parent``. Run the server parent on a stable host with a
stable external NVFlare federation endpoint.

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

Mount sources must be absolute paths outside the prepared kit and runtime
workspace, and must exist on the compute nodes that can run the job. Bare mode
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
``cpus_per_node``, ``mem_per_node`` (MiB), ``time``, and
``pending_timeout``. A job image requires normal BYOC authorization and takes
precedence over the study and site images. Job and study images are rejected on
any site whose effective sandbox is ``none``.

Multi-node jobs require effective ``sandbox: none`` and must not specify an
image. A positive multi-node ``num_of_gpus`` requires ``gpus_per_node``;
whenever both are supplied, ``num_of_gpus`` must equal
``nodes * gpus_per_node``.
The application owns any multi-node process fan-out.

Security and Operations
=======================

The worker-to-parent internal channel is clear TCP, matching the Kubernetes
launcher. Use it only on a trusted or isolated site network. This does not
change the configured security of the external NVFlare federation channel.

Working accounting is mandatory. The parent refuses to start if ``sacct`` is
unavailable. A later scheduler or accounting outage leaves affected jobs
non-terminal and retries; it never assumes that a missing observation means a
job has stopped.

NVFlare stores the deployment identity and transient launch artifacts under
``workspace_path/.nvflare_slurm``. The live parent removes a job's artifacts
after launch failure or terminal completion. User abort and pending timeout
verify ownership before ``scancel``; normal framework shutdown terminates
running handles through the same path before the Slurm launcher closes launch
admission.

Job output is ``<run-dir>/slurm-<slurm-job-id>.out``. Use ``squeue`` for live
state and ``sacct`` for completed jobs. Preserve the complete workspace and
relevant scheduler records for investigation.

After a parent crash, the launcher does not cancel surviving allocations or
remove their job artifacts; this matches the Docker and Kubernetes launchers.
A leftover job directory blocks relaunch of that job ID until an operator
verifies that no old allocation uses it and removes the directory.

An ``sbatch`` timeout fails the FL dispatch even if Slurm accepted the job.
Artifact removal prevents a pending allocation from starting unless the same
job ID is relaunched before it starts.

To change ``workspace_path``, stop the parent and use a new empty path. To keep
existing data, move the complete workspace, prepare again with the new path,
and stage the newly prepared kit there. Moving only run directories or only the
``.nvflare_slurm`` control data breaks the deployment identity.

Server-job queue and startup time must fit within the client runner-sync timeout
(60 seconds by default). Provide prompt server capacity or increase
``max_runner_sync_timeout`` when the site cannot meet that bound.

Before production use, test on the target cluster:

#. Successful, failed, timed-out, pending-aborted, and running-aborted jobs.
#. Parent restart with a live job and re-prepare/re-stage data preservation.
#. Compute-node connectivity to the parent and multi-node collectives when used.
#. Slurm accounting, association, QOS, partition, cgroup, and GPU enforcement.
#. The selected backend's filesystem view, environment, secret handling, exit
   status, and CPU/GPU device visibility.

Do not describe Apptainer as a production sandbox until those isolation checks
pass on the deployment cluster.
