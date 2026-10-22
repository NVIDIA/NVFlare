.. _docker_task_execution:

Docker Task Worker Setup and Test Procedure
===========================================

Use this procedure for the Docker task-worker candidate: a Docker client parent
(CP), a CPU-only client job (CJ), and disposable CPU or GPU task workers.
Deployment qualification is still in progress. Passing a host or image check
does not establish that an NVFlare task job passed.

Start with the :github_nvflare_link:`Docker runtime example <examples/docker>`
for image builds, provisioning, runtime-kit preparation, startup, admin
connectivity, and submission. The additional task profile is described in the
:github_nvflare_link:`Docker task execution contract <docs/design/docker_task_execution.md>`.
The standard ``nvflare deploy prepare`` command currently prepares resident job
launchers; task-worker settings must be applied to the prepared client kits
before their parents start.

Keep One Prepared Host
----------------------

Use a dedicated Linux Docker daemon and a non-root account that can access its
socket. Record the lease/host identity, daemon ID, driver version, approved GPU
UUID partition, account UID/GID, and workspace path once. All participating CPs
must use this daemon and the approved partition's admission protocol. Exclude
uncoordinated host processes, Process jobs, and other schedulers from that GPU
partition.

Before building or provisioning, run these checks from the account that will
start the parent kits:

.. code-block:: bash

   test "$(id -u)" -ne 0
   id
   docker version --format '{{.Server.Version}}'
   docker info --format '{{.ID}} {{.OSType}} {{.DockerRootDir}}'
   nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
   nvidia-ctk --version

The GPU checks apply to a GPU campaign. A CPU-only host does not need the NVIDIA
driver or Container Toolkit. Resolve a failed check at this stage. Preserve the
working driver, toolkit, Docker installation, prepared workspace, and build
cache across candidate source updates.

Freeze Source Before Building
-----------------------------

Use one candidate checkout and its supplied source manifest. A base commit
alone does not identify a candidate with an uncommitted patch. An archive of
``HEAD`` omits those changes and untracked implementation files.

.. code-block:: bash

   cd /path/to/candidate/NVFlare
   export NVFL_CANDIDATE_ROOT="$PWD"
   export PYTHONPATH="$NVFL_CANDIDATE_ROOT"
   git rev-parse HEAD
   sha256sum --check /path/to/candidate/source-manifest.sha256
   python - <<'PY'
   import os
   from pathlib import Path
   import nvflare

   actual = Path(nvflare.__file__).resolve()
   expected = Path(os.environ["NVFL_CANDIDATE_ROOT"]).resolve() / "nvflare"
   assert actual.is_relative_to(expected), (actual, expected)
   print(actual)
   PY

Keep ``PYTHONPATH`` set for provisioning and probe export. A shared virtual
environment may contain an editable install pointing to another checkout.
Preserve the verified candidate archive and manifest outside the lease.

Build Images and Retain the Dependency Cache
--------------------------------------------

The existing example builds both parent and job images from the checkout:

.. code-block:: bash

   (cd examples/docker && bash build_docker.sh)
   export NVFL_PARENT_IMAGE="$(docker image inspect nvflare-site:latest --format '{{.Id}}')"
   export NVFL_WORKER_IMAGE="$(docker image inspect nvflare-job:latest --format '{{.Id}}')"
   docker image inspect "$NVFL_PARENT_IMAGE" "$NVFL_WORKER_IMAGE" \
     --format '{{.Id}} {{json .RepoDigests}}'

Use the recorded immutable IDs in the prepared configuration and exported job
metadata. ``examples/docker/Dockerfile.nvflare-job`` installs heavy dependencies
before copying application source, so a source-only change can reuse that layer.
Keep the build context, Dockerfile, base image, build arguments, and package
metadata stable when they have not changed. Avoid ``--no-cache`` and build-cache
pruning during a campaign. A changed dependency manifest legitimately requires
a new dependency layer.

Both images must contain the candidate. In particular, a CP service fix requires
the new parent image and a parent restart; updating only the worker image leaves
the old CP code running. Record Python, NVFlare, Docker SDK, PyTorch, and CUDA
versions from the resulting images. The parent needs the Docker SDK; the probe
job image needs CUDA-capable PyTorch for the GPU campaign.

Check the Images Before Submitting a Job
----------------------------------------

Run image checks with the same numeric user as CP and the worker. The generated
``start_docker.sh`` uses the invoking account's UID/GID and adds the daemon socket
group. The prepared workspace must be writable by that account.

.. code-block:: bash

   export NVFL_RUN_USER="$(id -u):$(id -g)"
   docker run --rm --network none --user "$NVFL_RUN_USER" \
     --entrypoint /usr/local/bin/python "$NVFL_PARENT_IMAGE" \
     -c 'import docker, nvflare; print(nvflare.__file__, docker.__version__)'

   docker run --rm --network none --user "$NVFL_RUN_USER" \
     -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=none -e CUDA_VISIBLE_DEVICES= \
     --entrypoint /usr/local/bin/python "$NVFL_WORKER_IMAGE" \
     -c 'import os, torch; assert os.getuid() != 0; assert torch.cuda.device_count() == 0; print(torch.__version__)'

For GPU preflight, choose one approved physical UUID and run the following only
while that partition is idle, before task execution starts:

.. code-block:: bash

   export NVFL_TEST_GPU=GPU-replace-with-approved-full-uuid
   docker run --rm --network none --user "$NVFL_RUN_USER" \
     --gpus "device=$NVFL_TEST_GPU" -e HOME=/tmp \
     -e NVIDIA_VISIBLE_DEVICES="$NVFL_TEST_GPU" -e CUDA_VISIBLE_DEVICES=0 \
     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
     --entrypoint /usr/local/bin/python "$NVFL_WORKER_IMAGE" \
     -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; x=torch.ones(16777216, device="cuda"); x.square_().add_(1); s=x.sum().item(); torch.cuda.synchronize(); assert s == 33554432.0; print(torch.__version__, torch.version.cuda, s, torch.cuda.max_memory_allocated())'

Repeat GPU preflight for every UUID that the site partition will use. A
successful ``nvidia-smi`` or device-count check alone does not verify the probe's
allocation and CUDA kernels. Inspect the imported source path and verify
candidate ``nvflare/`` file hashes inside both images against the same manifest;
ignore manifest entries for tests/docs that are not included in the images.

Prepare and Start the Task Profile
----------------------------------

Follow the Docker example's provisioning and ``nvflare deploy prepare`` steps.
Set ``parent.docker_image`` to the frozen parent image ID in the preparation
config. Before startup, apply the task contract's configuration to each prepared
client's canonical ``local/resources.json`` or its ``resources.json.default``
fallback:

* Retain exactly one matching ``ClientDockerJobLauncher`` and its existing
  daemon/network settings. Add its ``task_worker_config`` with the frozen worker
  image ID, image Python path, worker environment, and approved GPU partition.
* Select ``DockerTaskLauncher`` in ``task_execution.launcher``. Worker environment
  belongs to ``task_worker_config.environment``; Process environment forwarding
  does not configure this profile.
* Keep the same non-root UID/GID and persistent job path in CP, CJ, and workers.
  The daemon-host bind source must exist and be writable; CP derives worker
  mounts from its existing launcher configuration.
* Use Docker-native ``launcher_spec[site]["docker"]["num_of_gpus"]`` for task GPU
  requests. Positive portable/legacy CP ResourceManager GPU requests are rejected
  because they otherwise reserve GPUs for the whole job. Explicit zero overrides
  a GPU site default. Server and CPU probe requests must explicitly use zero.

Start the prepared server/client kits using ``startup/start_docker.sh`` and
confirm registration with ``nvflare system status`` as shown in the example.
The task CJ receives neither GPU devices nor a Docker socket. Keep parent
daemon access confined to CP; no additional socket mount belongs in CJ or worker
configuration.

Run the Application, Then Lifecycle Checks
------------------------------------------

Export the same probe for CPU and CUDA using the frozen worker image:

.. code-block:: bash

   python tests/integration_test/data/docker_task_execution/export_probe.py \
     --image "$NVFL_WORKER_IMAGE" --device cpu --clients 1 \
     --lifetime task --output /tmp/nvflare-task-probes
   python tests/integration_test/data/docker_task_execution/export_probe.py \
     --image "$NVFL_WORKER_IMAGE" --device cuda --gpu-count 1 --clients 1 \
     --lifetime task --output /tmp/nvflare-task-probes

Submit the CPU job first with ``nvflare job submit`` using the prepared admin
connectivity. Verify its terminal status and per-attempt logs before submitting
the GPU job. The probe uses two rounds, unchanged model parameters, and real
tensor computation. CPU must see zero GPUs. GPU logs must show CUDA allocation
and kernels on the admitted device. Each task assignment needs its own native
container identity; PID 1 alone does not distinguish worker containers.

Record candidate/source hashes, image IDs, job/site/task/attempt IDs, container
IDs, resource manifests, terminal status, and logs for each row. A resident
example result or an earlier candidate's result cannot qualify this task
candidate. Record setup activity separately from submitted/running/terminal jobs.
Continue with the contract's cancellation, failure, saved-result replay,
active-reader, checkpoint, settlement, and GPU contention/reuse checks. Preserve
diagnostics and payload retention evidence before cleaning up owned containers.

Correct Four-Row Serial Control
-------------------------------

The resident GPU control must not compete with an active task partition. Prepare
all four jobs from one candidate and the same compute script, using two distinct
physical UUIDs rechecked by the lease coordinator on existing authorized
capacity:

.. code-block:: bash

   python -m tests.integration_test.data.docker_task_execution.serial_matrix \
     --image "$NVFL_WORKER_IMAGE" \
     --site-gpu "site-1=$NVFL_GPU_SITE_1" --site-gpu "site-2=$NVFL_GPU_SITE_2" \
     --output /tmp/nvflare-serial-matrix

This exports CPU job, CUDA job, CPU task and CUDA task jobs with one server,
two sites and two rounds. It writes public ``phase-policies.json`` and a
``campaign.json`` whose rows start as ``requested``. It never contacts Docker,
starts a job, copies a private kit or changes a site's live configuration.

Use the existing prepared kits on the owned host in this order:

1. Recheck lease ownership, availability and unrelated workloads. Confirm prior
   jobs are terminal and native task workers and GPU reservation records are
   absent. Preserve the earlier failure's config, job ID and diagnostics.
2. Stop only the campaign's owned parents using supported shutdown. Apply the
   resident phase to both canonical client resource files, restart the parents,
   and verify their live source imports, image IDs and canonical configuration.
3. Run CPU job, then CUDA job to terminal with artifact/result and native cleanup
   checks. Resident site policy assigns one distinct UUID to each site through
   ``default_job_container_kwargs.device_requests``; no task partition is active
   in this serial phase. The GPU resident job intentionally omits a GPU count:
   a count would replace site-owned UUID requests with ``Count=N``. CPU and
   server requests explicitly use zero and suppress the inherited GPUs.
4. After supported shutdown and another no-job/no-worker/no-reservation check,
   apply the task phase and restart the parents. It removes resident defaults
   and restores the shared approved exclusive partition. Run CPU task, then
   CUDA task using the same source and images. GPU count one is bound by CP and
   admitted through normal task reservations; the task CJ remains GPU-free.
5. Verify terminal results, per-site compute records and distinct native worker
   identities, artifact availability, and no owned worker/GPU/reservation
   residue. All four rows require new recorded job IDs and verified-passed
   results; a skipped, failed or unexplained row leaves the campaign incomplete.

The fixture's ``apply_site_policy(site_config, policy)`` returns a modified copy
of the existing canonical client resource dictionary. It preserves other
components, network settings, limits, worker environment and security settings.
The coordinator can apply it locally after stopping the parent; do not export
the private prepared kit. Keep the generated ``resources.json.default`` and use
a canonical ``resources.json`` override. Changing a partition in configuration
does not release an existing native reservation, and a running CP retains its
old policy until restart.

Track the shared job-directory credential boundary separately: a task worker
can access ``job_cert/job.key`` through the same-UID job bind when that file is
present. This fixture does not change mounts or credentials, and passing the
four compute rows does not establish credential-free filesystem isolation.

Respond to a Failure Without Restarting Setup
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Failure
     - Next action
     - Reuse
   * - Driver/toolkit or CPU/CUDA image preflight
     - Resolve the specific host/runtime/image dependency before product launch.
     - Verified source and unrelated prepared assets.
   * - Source or imported-package mismatch
     - Correct checkout, archive, ``PYTHONPATH``, or image source; recheck hashes.
     - Host runtime and dependency layers.
   * - Kit permissions, mounts, addresses, or registration
     - Correct the affected prepared kit and restart its owned parent.
     - Candidate source, images, and host runtime.
   * - Product exception after job submission
     - Preserve the job and trace; reproduce, fix, and test the failing boundary.
     - Host runtime, dependencies, and unaffected configuration.
   * - Source-only product fix
     - Freeze the new source, rebuild affected source layers, restart affected
       parents, and submit a new identified run.
     - Lease, driver/toolkit, dependency cache, and persistent evidence.

For example, Docker task replies must transmit ``TaskExecutionPhase.value`` as
a primitive string. The real Cell/FOBS regression runs with enum
auto-registration disabled. A serialization failure is a product issue;
reinstalling Docker or downloading PyTorch again does not repair that boundary.
