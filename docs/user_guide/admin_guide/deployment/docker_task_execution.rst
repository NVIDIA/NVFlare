.. _docker_task_execution:

Docker Task Worker Setup and Test Procedure
===========================================

Use this procedure for the Docker task-worker candidate: a Docker client parent
(CP), a client job (CJ) with its own budget, and disposable CPU or GPU workers.
The CJ defaults to CPU coordination with no GPUs.
For assignment handling, receipts, payload retention and settlement, see the
:github_nvflare_link:`shared task execution design <docs/design/task_worker_lifecycle.md>`.
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
* Existing ``resource_spec`` and native launcher fields describe compute: CJ in
  job mode and task workers in task mode. Give GPU workers CPU and memory too.
  Use ``cj_resource_spec`` for the separate task-mode CJ budget; export/submission
  derives its applicable sites from actual client configs. Omitted CJ settings
  use the launcher's trusted ``cj_resources`` defaults: 0.25 CPU, 512Mi, zero GPUs.
* Worker portable CPU/memory requirements become Docker cgroup limits; omitted
  fields use ``task_worker_config.nano_cpus`` / ``mem_limit`` defaults of one CPU
  and 1Gi. These are limits, not exclusive host CPU reservations. GPU requests
  can use portable counts or Docker-native counts, subject to shared conflict
  checks. Set server GPU requirements to zero. Re-export/resubmit old task jobs
  whose metadata would still reserve compute GPUs for the whole CJ lifetime.

Start the prepared server/client kits using ``startup/start_docker.sh`` and
confirm registration with ``nvflare system status`` as shown in the example.
The default task CJ receives no GPU devices. An explicit independent CJ GPU
budget holds CP-admitted UUIDs until CJ removal is confirmed; workers acquire
and release their own devices per task. Keep daemon access confined to CP;
no Docker socket belongs in CJ or worker configuration. Custom RM and positive
per-GPU memory claims that this profile cannot enforce are rejected.

Task CJs reject Docker control environment variables, host or other-container
PID/IPC namespaces, extra capabilities, and privileged/device/socket inheritance.
Keep CP daemon credentials out of approved datasets and study-secret mounts.
The task profile fixes ``cap_drop=ALL`` and ``no-new-privileges`` for CJ.
Worker creation also fixes ``init=False`` so the authorized Python worker stays
PID 1 even when the Docker daemon has ``init: true`` as its default.

``DockerTaskLauncher`` accepts ``cleanup_timeout`` (default 10 seconds) and
``poll_interval`` (default 0.05 seconds). Cancellation retries temporarily
unsettled replies within that budget. One absolute deadline covers all streaming
RPC phases, including continued progress; late replies are rejected. Align ``cleanup_timeout`` with the site's ``task_execution.settlement_timeout``
when configuring a longer cleanup allowance. Expiry retains the supervisor
fence; later CP cleanup does not authorize another physical attempt.

CP retains up to 10000 attempt identities per job by default, including rejected
launches and cancel-before-launch fences. Configure
``task_worker_config.max_retained_attempts`` in the trusted CP policy. The CJ
``DockerTaskLauncher`` has a separate ``max_retained_attempts`` argument with the
same default. Exhaustion closes new admission and preserves existing replay
fences and handle cleanup; it never evicts old entries to admit new attempts.

Resident and task Docker launchers publish the created job handle to the client
executor before waiting for RUNNING. Admin and heartbeat abort can therefore
reach a never-running CJ during that wait, including abort intent recorded
before publication. An outstanding Docker create/start API call still requires
its outcome to be reconciled; publication does not cancel the API call itself.

CP retains the task job before native creation and monitors cleanup after CJ
exit. A termination request can return while Docker is unavailable; the job
remains pending cleanup until CP confirms owned CJ/worker removal and GPU
reservation release. Abort stops the owned CJ independently of uncertain worker
or GPU-reservation cleanup. Its container record and job session remain retained
until physical settlement is confirmed; then CP removes the finished session
and rejects its old capability. Keep CP running during recovery and preserve the
job workspace. An unexpected missing container or lost API reply requires
reconciliation rather than manual capacity release. The earlier four-row
compute evidence predates these authority/lifecycle changes; qualify the new
source and images again.

Live Worker Connection
----------------------

Keep CJ and task workers on the launcher's same user-defined Docker network.
Docker's built-in ``bridge``, ``host``, ``none`` and ``container:...`` modes are
unsupported for task workers. CP supplies the actual CJ container DNS name via
``NVFLARE_DOCKER_TASK_LISTENER_HOST``; it is consumed during START_RUN and is
not a worker environment setting. The listener inherits the site's internal
scheme and security and allocates a fresh port. Permit worker-to-CJ traffic
inside that network. Local mounted data remains the initial data-access profile.

Workers must connect and complete HELLO before compute. In secure mode, inspect
bootstrap credential paths without printing certificate/key contents: the public
CA must be mounted read-only, and both client and server TLS roles must use the
job certificate/key under the job bind. Site private keys and CP Docker authority
must remain absent. Metrics travel over the live Cell session; no analytics
spool or post-exit metric replay is used. See
:github_nvflare_link:`worker Cell session design <docs/design/task_worker_cellnet_session.md>`
for the shared protocol.

Persistent Job State and Restart Boundaries
-------------------------------------------

The accepted default site layout uses one private CP workspace store and one
separate jobs-state store. CP accesses both. Each resident or task CJ, and its
task workers, receives only its own job directory from the jobs-state store,
plus approved startup/configuration and dataset bindings. Keep site private
keys and broad Docker authority in CP. The number of stores does not grow with
the number of jobs or task attempts.

The current Docker launcher has a single daemon-host ``workspace`` root and
derives each job's bind source from it. Containers already receive individual
job binds, but a separate jobs-state root is not configurable yet. Implementing
the two-store default remains alignment work before readiness; do not add
unsupported storage settings or expose CP's full workspace to a CJ/worker.

For both ``execution_lifetime="job"`` and ``execution_lifetime="task"``, write
required checkpoints and recovery metadata under the persistent job path
``/var/tmp/nvflare/workspace/<job_id>``. Container removal keeps that bind's
files, including retained task artifacts. The container's writable layer and
workspace-root tmpfs remain ephemeral; root-level ``snapshot-storage`` or
``jobs-storage`` directories outside the job bind do not provide persistence.
Verify the application's actual save paths for both lifetimes during deployment
qualification.

Resident Docker CJs participate in main's automatic CJ recovery after confirmed
container exit when the shared policy permits it. CP retains the job workspace,
certificate and resource reservation, refreshes launch tokens, and makes each
replacement handle abortable during startup. ``max_cj_restart_failures`` in
``local/application.conf`` defaults to three consecutive failed replacement
initializations; READY resets the count. Applications restore their own
checkpoints. CP and the server must remain running; intermediate CJs are excluded.

Task-mode Docker CJs opt out of automatic restart. Server redelivery retains the
same task/attempt ID, and a new CJ cannot safely reuse its retained artifact
directory or CP replay fences. This also applies when CJ staged files before
its worker launch reached CP. CP still confirms owned worker cleanup and GPU
release before reporting failure. Retained files alone do not implement task
resume, and CP restart adoption remains unsupported for both lifetimes. Preserve
required state and unresolved cleanup evidence until reconciliation and the
existing artifact/reader retention rules allow removal. The job bind still has
the job-credential exposure described under qualification below.

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

Before counting a row as passed, retrieve its metadata from the server using
the same study and prepared admin connectivity used for submission:

.. code-block:: bash

   nvflare job meta "$NVFL_JOB_ID" --format json > "$NVFL_JOB_ID.server-meta.json"

For a nondefault study, add ``--study <study>``. Compare the resource fields in
that response with the exported metadata. Preserve literal ``@default`` and
``@ALL`` keys and all site overrides; ``default`` is a different key. For the
task probes, check both ``resource_spec`` and ``cj_resource_spec``, including the
server's explicit zero-GPU override and the derived task-site entries. An export
alone cannot show whether submission preserved the defaults. Server admission
reads the client's literal ``execution_lifetime`` without resolving client-only
environment variables or opening client HOCON includes. The client receives its
original configuration; CP and CJ check the locally resolved lifetime again
before constructing application components.

While each owned CJ and task worker still exists, capture its native settings.
Use its recorded container ID and collect only these fields, keeping session
capabilities and credentials out of the evidence:

.. code-block:: bash

   docker inspect --format '{"id":{{json .Id}},"state":{{json .State.Status}},"nano_cpus":{{json .HostConfig.NanoCpus}},"memory":{{json .HostConfig.Memory}},"devices":{{json .HostConfig.DeviceRequests}}}' \
     "$NVFL_CONTAINER_ID" > "$NVFL_CONTAINER_ID.resources.json"

For the exported task probes above, every client must have these live limits:

.. list-table::
   :header-rows: 1

   * - Runtime
     - NanoCpus
     - Memory (bytes)
     - GPU device requests
   * - CPU or CUDA task CJ
     - 250000000
     - 1073741824
     - None
   * - CPU task worker
     - 1000000000
     - 1073741824
     - None
   * - CUDA task worker
     - 1000000000
     - 1073741824
     - One CP-admitted UUID from the approved partition

The probe explicitly sets its CJ memory to 1Gi, overriding the site's 512Mi
CJ default. Check every task attempt before CP removes its container, and retain
its resource manifest and compute logs. The probe sends a scalar metric after
its real CPU/CUDA kernels; ``live_metric_sent=true`` records acknowledgement
before result publication. Require CJ-side live metric evidence and successful
pre-compute HELLO. Under secure deployment, also verify unreachable CJ
DNS/listener, missing job credentials and another job's certificate fail before
compute. Local Cell tests alone do not qualify Docker DNS/mounts. Successful CUDA
computation alone does not establish that CPU/memory limits survived submission.
Missing metadata or native-setting evidence leaves the row unqualified.

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
   admitted through normal task reservations. The exported task probe requests
   one CPU and 1Gi for compute, independently of a 0.25 CPU/1Gi/zero-GPU CJ.
   Inspect both budgets in native container settings and verify worker GPU reuse
   while CJ remains alive.
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
