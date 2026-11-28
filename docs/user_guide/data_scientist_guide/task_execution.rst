.. _task_execution:

Run Tasks in Slurm or Kubernetes Workers
========================================

NVFlare 2.10 supports running each client task attempt in a fresh Slurm
allocation or Kubernetes Pod. A client parent (CP) starts a client job (CJ)
at each site. With ``execution_lifetime="task"``, CJ coordinates federation
communication while CP launches and cleans up the compute workers. The server
workflow continues across tasks and rounds.

When to use task lifetime
-------------------------

* Release worker GPUs while other sites finish their tasks or the server
  aggregates results. The scheduler can reuse those GPUs after worker cleanup
  is confirmed. The coordinator remains available on its own resource budget.
* Let sites acquire GPUs at different times. A synchronous round still waits
  for its required results, but sites can complete their tasks without holding
  GPU allocations simultaneously for the whole job.
* Give each Slurm task its own queue wait and wall-time lease. An early site's
  job-long GPU allocation can otherwise expire while another site is still
  queued. CP and CJ still need to outlive the federation job and cleanup, and
  queueing and execution must fit the configured deadlines.

Fresh workers repeat startup, imports, data loading, and checkpoint I/O. Compare
that overhead with the time and resources retained between tasks by your job.

.. list-table:: Client execution lifetimes
   :header-rows: 1
   :widths: 24 38 38

   * - Behavior
     - Job lifetime (default)
     - Task lifetime
   * - Compute location
     - The resident CJ executes successive tasks.
     - A fresh worker executes each task attempt.
   * - Compute resources
     - CJ retains its CPU, memory, and any GPUs for its lifetime.
     - Each worker has its own allocation or Pod. CJ has a separate budget.
   * - Local application state
     - Objects can remain in memory across tasks.
     - State needed by the next worker must be checkpointed and restored.
   * - Site setup
     - The ordinary job launcher starts CJ.
     - Site IT also configures a compatible task launcher and worker access.

Site IT enables the backend
---------------------------

At **each participating client site**, Site IT configures the prepared CP
workspace. Copy ``local/resources.json.default`` to ``local/resources.json``
when an active override does not exist. Keep one supported client job launcher
and add ``task_execution.launcher`` to that same file. Preserve the site's other
components, then restart or roll out CP to load the settings. Reapply the
override if deployment preparation regenerates the workspace.

Each site manages its own storage and scheduler access. Shared task artifacts
need coherent storage within that site. Participating sites can use separate
filesystems and keep their datasets local.

``local/study_runtime.yaml`` supplies optional, site-owned study settings such
as environment variables and dataset locations. It is separate from the
backend configuration in ``resources.json``. Researchers select the study
approved by Site IT and use its worker-visible paths.

Slurm sites
~~~~~~~~~~~

Site IT follows the :doc:`Slurm task deployment guide
<../admin_guide/deployment/slurm_task_launcher>` to configure the launcher pair,
account, partitions, worker interpreter, and resource defaults.

* **Placement.** CP can run on a login/service host or inside a CPU allocation
  that permits further Slurm submissions. A Process CJ shares CP's host and,
  when applicable, its allocation. A native Slurm CJ gets its own allocation.
  A Docker site can use Docker CP and Docker CJ with native Slurm workers.
* **Scheduler access.** Verify ``sbatch``, ``squeue``, ``sacct``, and ``scancel``
  under CP's actual runtime UID. A Docker CP needs compatible Slurm clients,
  site-managed authentication and configuration mounts, and access to the
  scheduler. Its Docker host must be an authorized cluster submit host.
* **Storage and identity.** Use the same canonical workspace path and UID on
  the submit host and compute nodes. Workers need their Python interpreter,
  dependencies, application files, and datasets at the configured paths.
  ``shared_filesystem`` and ``cgroup_enforced`` must both be ``true`` after
  verifying the filesystem and scheduler enforcement.
* **Worker connectivity.** GPU/CPU worker nodes must reach CJ's dynamic Cell
  listener. For a Process or Docker CJ, set ``worker_listener_host`` to the
  compute-reachable CP host. Docker task CJs use Linux host networking.
  Verify the route and firewall from a real allocation.

For a native Slurm CJ, the job and task launchers must have the same
``workspace_path``. Configure the CJ with ``sandbox: none`` and no image.
Workers use one node and require CPU and memory requests, including GPU
workers. This profile rejects ``additional_node_command``, worker images in
``launcher_spec``, and ``mem_per_gpu_in_GiB`` requests.

For a Docker host outside the compute cluster, verify scheduler access,
worker-to-CJ routing, and shared storage separately. A reachable login node or
a successful Docker launch establishes only part of that deployment.

Kubernetes sites
~~~~~~~~~~~~~~~~

Site IT follows the :doc:`Kubernetes task deployment guide
<../admin_guide/deployment/kubernetes_task_launcher>` to configure
``ClientK8sJobLauncher`` and ``K8sTaskLauncher`` at each site.

* **Placement.** Configure the namespace, coordinator image selection, worker
  image, service accounts, and resource defaults in the prepared site kit.
  CP uses Kubernetes API permissions to create and clean up workers.
* **Storage and identity.** Configure separate CP workspace and jobs PVCs in
  ``helm_chart/values.yaml``. Match the launcher PVC names, mount paths, and
  nonroot UID/GID in ``local/resources.json``. Verify concurrent access when
  CJ and workers can be scheduled on different nodes.
* **Dataset access.** Verify a real sample in a worker Pod using the intended
  image and UID/GID. A dataset mounted only in CJ also needs a worker mount.
  Site IT configures worker dataset PVCs and Secret references through the
  selected study in ``local/study_runtime.yaml``.
* **Worker connectivity.** Allow worker-to-CJ Pod traffic on the dynamic Cell
  listener through cluster routing and NetworkPolicy. Workers also need
  access to the configured images and shared job artifacts.

After changing the prepared kit, stage it with ``nvflare deploy k8s stage``
and apply the Helm upgrade. Verify the settings mounted in the running CP.

Researcher selects task lifetime
--------------------------------

1. Confirm with Site IT which sites support task lifetime, the approved study
   and images, and the worker-visible dataset paths.
2. Check :ref:`recipe_execution_lifetime` for supported recipes. Task workers
   support ``ClientAPIExecutor``, ``StatisticsExecutor``, and ``NPTrainer``.
   Client API scripts must use in-process execution
   (``launch_external_process=False`` when constructing a recipe).
3. Select task lifetime before exporting. For an existing recipe, use the
   common client configuration helper:

   .. code-block:: python

       recipe.add_client_config({"execution_lifetime": "task"})
       recipe.export("/tmp/task-jobs")

   Recipes listed with constructor support also accept
   ``execution_lifetime="task"`` directly. With the Job API, pass
   ``execution_lifetime="task"`` to ``FedJob`` or call
   ``job.set_execution_lifetime("task", target="site-1")`` for one site.
   Handwritten client configs set ``"execution_lifetime": "task"`` at the
   top level of ``config/config_fed_client.json``.

4. Size workers with portable ``resource_spec`` or native ``launcher_spec``
   compute fields. Use ``cj_resource_spec`` for CJ's separate budget. For
   example, merge these fields into the exported ``meta.json``:

   .. code-block:: json

       {
         "resource_spec": {
           "@default": {"num_of_cpus": 4, "memory": "8Gi", "num_of_gpus": 1},
           "server": {"num_of_cpus": 1, "memory": "2Gi", "num_of_gpus": 0}
         },
         "cj_resource_spec": {
           "@default": {"num_of_cpus": 1, "memory": "2Gi", "num_of_gpus": 0}
         }
       }

   Each worker requests four CPUs, 8 GiB, and one GPU. CJ requests one CPU,
   2 GiB, and zero GPUs. Set worker ``num_of_gpus`` to ``0`` for CPU compute.
   Size CJ for model transfer, filters, and result processing as well as
   coordination, especially when exchanging large models.
   A Process CJ has no enforced coordinator budget. Omit a nonzero
   ``cj_resource_spec`` for that site. Use site-specific entries when the
   federation has different coordinator types. Kubernetes jobs also need a
   CJ image from job metadata or the site's study settings.

5. Preserve the application state needed by the next worker and submit the
   exported job through the normal NVFlare CLI. Local simulation uses job
   lifetime. Configuring task workers at a site also permits ordinary
   job-lifetime applications through that site's job launcher.

Task lifetime rejects SCAFFOLD workflows and cross-site evaluation that asks
clients to submit local models. FedCE's retained local state also requires
job lifetime. Worker component configs reject ``execution_scope`` and
``component_dependencies``. Evaluation of a server-provided global model is
supported. See the recipe compatibility table for additional restrictions.

Preserve training state between workers
---------------------------------------

The received global model and the previous local training checkpoint serve
different purposes. Save and restore optimizer, scheduler, random-number,
and data-position state when the algorithm needs them across tasks.

After ``flare.init()``, Client API scripts read
``flare.system_info()["task_checkpoint"]``. Its ``input`` is the previous
successfully committed checkpoint path, or ``None`` initially. Write the new
candidate to ``output`` before completing the task. The runtime promotes it
on successful local task completion. The application owns the checkpoint
format and restoration logic.

The :github_nvflare_link:`Hello PyTorch client
<examples/hello-world/hello-pt/client.py>` demonstrates native PyTorch
checkpoint save and restore. A file readable by CJ still needs to be readable
by the worker's runtime identity and mount layout.

Export, submit, and verify a small job
--------------------------------------

After Site IT enables both sites, use the
:github_nvflare_link:`Hello PyTorch example <examples/hello-world/hello-pt>`
from a checkout matching the deployed version. Install its declared
dependencies in the export environment and ensure the site runtimes contain
the application dependencies. From the example directory, export a
two-client, three-round job with synthetic data:

.. code-block:: bash

    export JOB_ROOT=/tmp/nvflare-task-jobs
    python job.py --export --export-dir "$JOB_ROOT" \
      --execution_lifetime task --dataset synthetic --n_clients 2 --num_rounds 3

Use the site's agreed resource budgets, study, and images before submission.
The example defaults to CPU computation. For CUDA qualification, use the
backend guide's GPU probe and verify actual CUDA work in each worker.

.. code-block:: bash

    export ADMIN_KIT=/path/to/admin/startup-kit
    nvflare job submit -j "$JOB_ROOT/hello-pt" --startup-kit "$ADMIN_KIT"
    export JOB_ID='job-id-from-submit'
    nvflare job wait "$JOB_ID" --timeout 1800 --startup-kit "$ADMIN_KIT"
    nvflare job download "$JOB_ID" -o "$JOB_ROOT/results" --startup-kit "$ADMIN_KIT"

Confirm ``FINISHED:COMPLETED``, contributions from both sites, the expected
round count, and a valid final model. Check restoration of local state on
successive training tasks. Then have Site IT verify worker release using
Slurm accounting or Kubernetes shutdown evidence, and run another allocation
to confirm capacity reuse. A successful training result and confirmed resource
release are separate observations.

If a worker cannot open data, check its resolved path, UID/GID, and mounts.
If it cannot connect to CJ, check the advertised address, dynamic listener,
and compute-node routing. If it stays queued, inspect scheduler admission,
resource requests, and queue deadlines. CP must stay alive through worker
cleanup. Follow the backend guide's reconciliation procedure after CP loss.
