:orphan:

***************************
What's New in FLARE v2.10.0
***************************

Highlights:

- **Security and identity** — optional single sign-on for administrators
  through an OpenID Connect identity provider (step-ca), optional
  certificate-derived study memberships, and per-job certificates (required
  in secure mode)
- **Memory and large models** — opt-in disk-backed FedAvg and FedProx
  aggregation, and disk-backed tensors for task-worker payloads
- **Resource efficiency** — opt-in task-scoped execution that lowers GPU
  waste, especially on HPC Slurm and Kubernetes clusters: GPUs are held only
  while a task trains and are released between tasks
- **Reliability** — automatic recovery of failed client jobs, such as after
  scheduler wall-time limits
- **Networking and deployment** — outbound HTTP proxy support, GPU
  reservation for Docker jobs, and a staged 2.10 upgrade path
- **Developer experience** — a rewritten Quick Start, an example catalog
  (``nvflare examples list`` / ``get``), progress logging, and refreshed
  self-paced chapters

Security and Identity
=====================

Single sign-on and certificate-derived study memberships are **optional**.
They are off unless you configure them, and existing static admin startup kits
and registry-based study access keep working unchanged. Per-job certificates
are different: in secure mode they are required, with no fallback to the site
certificate, so secure projects must be re-provisioned (see below).

Single Sign-On for Administrators (Optional)
--------------------------------------------

Administrators can sign in through their organization's OpenID Connect (OIDC)
identity provider. An admin startup kit can be provisioned without a bundled
key by setting ``admin_cert_provider``. At login, the built-in ``step_ca``
provider uses the ``step`` CLI, where step-ca performs the OIDC login and maps
the identity provider's claims to a certificate identity, organization, and
FLARE role, and issues an admin certificate chain rooted in the FLARE project
CA.

- FLARE validates the chain, validity period, key match, identity,
  organization, and role before use, then relies on the existing admin mTLS
  login, authorization, and job-signing paths. No OIDC tokens are introduced
  into the server or clients.
- Valid certificates are cached per OS user until they enter the renewal
  window, so the browser flow does not run for every CLI command.
- Static admin startup kits are unchanged and can coexist with provider-backed
  kits in the same ``project.yml``. A provider-backed kit cannot also use an
  externally supplied admin certificate.
- Requires the ``step`` CLI on the admin machine and a ``provisioner`` setting
  in the provider configuration.
- A downloadable step-ca template maps issuer-validated OIDC claims to
  certificate identity, role, and study memberships, and a certificate
  deployment guide is published.
- Admin login child processes are cleaned up after a timeout.

Per-Job Credentials and Studies
-------------------------------

- **Per-job certificates (required in secure mode).** In secure mode, each
  job process (server job and client job) uses its own short-lived credential
  issued by a job-signing CA, keeping the site's long-lived private key out of
  job processes. ``CertBuilder`` generates the job CA into the server startup
  kit by default, or the deployment can install its own (``external_job_ca``).
  A secure-mode server without a job CA refuses to deploy jobs; there is no
  fallback to the site certificate. Existing secure-mode projects must be
  re-provisioned, and the server and clients must run the same release.
  Non-secure deployments are not affected. Isolation does not apply to launch
  modes that cannot keep the site key out of the job process (for example the
  in-process launcher or Slurm ``sandbox: none``).
- **Certificate-derived study memberships (optional).** Administrators can
  access named studies through registry membership or, in addition, through
  membership carried in a validated certificate (one membership per URI SAN,
  up to 64 studies). Existing registry-based access is unchanged, and the
  registry stays authoritative for study existence and site enrollment.
- **Study environment settings.** Process jobs apply the selected study's
  ``env`` and ``secret_env`` from ``local/study_runtime.yaml``.

Provisioning and Certificates
-----------------------------

- Externally supplied endpoint certificates and server job CA.
- Public-root-only provisioning, and externally supplied file-based admin
  credentials.

Hardening
---------

- Dashboard role self-assignment and client property injection are blocked.
- PSI no longer logs site sizes or identities, and no longer echoes duplicate
  item values in validation errors.
- ``netrc`` can no longer override example API tokens.

Memory and Large Models
=======================

Disk-Backed Aggregation
-----------------------

The PyTorch ``FedAvgRecipe`` and ``FedProxRecipe`` accept
``enable_disk_aggregation=True``. Incoming updates remain disk references,
and aggregation reduces one tensor key at a time, so the server does not hold
the global model and all client updates in memory. The global model is
persisted as safetensors.

.. code-block:: python

   from nvflare.app_opt.pt.recipes import FedAvgRecipe

   recipe = FedAvgRecipe(
       min_clients=2,
       num_rounds=3,
       train_script="client.py",
       enable_disk_aggregation=True,
       initial_ckpt="/models/initial.safetensors",
   )

- Opt-in; the default keeps in-memory aggregation and persistence.
- Requires an initial safetensors checkpoint (a single file or a Hugging Face
  safetensors directory or index; directories and indexes need absolute
  server paths).
- Requires writable, disk-backed temporary and workspace storage with enough
  capacity for incoming updates and checkpoint slots. Memory still depends on
  tensor and batch size, concurrent transfers, and filesystem cache.
- Limits: custom ``model``, ``aggregator``, ``model_persistor``,
  ``model_locator``, or checkpoint filenames are rejected; persistence is
  weights-only with sequential checkpoint slots; native DCP, historical
  snapshots, full training-state or high-availability resume, and power-loss
  recovery are outside scope.
- Disk aggregation selects incoming-tensor offload itself; explicitly setting
  ``enable_tensor_disk_offload=False`` or a non-PyTorch
  ``server_expected_format`` with it is rejected.
- Incoming-tensor offload (``enable_tensor_disk_offload=True``, introduced in
  2.8) continues to be available when the global model stays in memory. It
  requires ``server_expected_format=ExchangeFormat.PYTORCH``.
- A deliberate download cancellation during workflow finalization is now
  distinguished from a genuine transfer failure, so a late cancelled download
  no longer terminates the server job.

Disk-Backed Task-Worker Payloads
--------------------------------

For PyTorch task-lifetime jobs, tensors in worker hand-offs are stored in
adjacent safetensors files, with tensor descriptors kept in the payload, so
the resident Client Job does not reconstruct full models. Incoming tensor
files are reused by hard link, and attempt files are released after terminal
receipt.

.. note::

   Behavior change: unfiltered Client Job input hooks receive local tensor
   references that support metadata and ``nbytes``, not tensor attributes such
   as ``.shape`` or ``.numel()``. Workers receive tensors.

Resource Efficiency
===================

Task-Scoped Execution
---------------------

**Purpose.** Federated training leaves GPUs idle. A site's GPUs sit
allocated while the site waits for the server to aggregate, for slower sites,
and for its own result upload, and that idle time counts against the site's
quota and keeps the GPU node unavailable to other work. By default the Client
Job holds its resources for the whole job, so on shared, scheduled clusters,
especially HPC Slurm and Kubernetes clusters, the GPU node cannot be released
between tasks.

FLARE 2.10 adds an opt-in ``execution_lifetime="task"`` setting that
separates the lifetime of computation from the lifetime of federation
coordination. A lightweight Client Job keeps the federation connection, task
acquisition, filters, privacy services, experiment tracking, and result
submission. Each assigned task runs in a fresh compute worker, and the
worker's GPU and other resources are released as soon as it has published its
output and physically exited, before the result is uploaded and before the next
round begins. The GPU node is therefore free between tasks. Job lifetime
remains the default.

**Slurm HPC.**

- **One allocation per task.** Each attempt gets its own Slurm allocation,
  sized for that task with a native ``launcher_spec`` or a portable
  ``resource_spec``, for both CPU and GPU compute. The wall-time limit now
  bounds one task, not the whole job.
- **A small coordinator budget.** The Client Job runs on a separate, small
  allocation selected by ``cj_resource_spec``, so GPUs are not held while the
  site waits.
- **Submission and cleanup are owned by the Client Parent**, which can run as
  a persistent service outside Slurm or inside an allocation. It submits,
  observes, and cancels task allocations and cleans up after them. A task's
  resources count as released only when allocation accounting confirms it, not
  merely because the Client Job allocation ended.
- Definitive scheduler rejections (partition, account, QoS, GRES, or resource
  policy) fail the task with the scheduler's message. Ambiguous submissions
  (timeouts, unknown output) keep cleanup ownership instead of guessing.
  ``task_shutdown_timeout`` (default 30 seconds) bounds shutdown, and an
  operator reconciliation command records an explicit decision when a
  submission cannot be resolved automatically.
- The existing ``ClientSlurmJobLauncher`` and ``ServerSlurmJobLauncher``
  behavior is preserved.
- Requirements and limits: shared absolute paths for code and interpreter,
  enforced Slurm cgroups, and identity-checked allocation accounting. The
  task-capable profile shares a workspace and UID and does not isolate
  credentials or filesystems between jobs, so job code must be trusted.
  Multi-node and DDP task execution, requeue adoption, and clusters without
  GRES are not supported.

**Kubernetes.** Task execution is available on Kubernetes (CPU and
GPU) through the existing ``ClientK8sJobLauncher``; trusted site policy
selects image, namespace, account, and resource defaults. A CPU Client Job can
stay connected while a task Pod owns the training GPUs, and the GPUs return to
the cluster when the Pod settles.

**Other benefits.** Between assignments, released GPUs can serve other
workloads on a shared cluster, and a wall-time limit now bounds one task rather
than the whole job. A crashed or killed worker does not take the federation
connection down with it.

**What this does and does not do.**

- Releasing resources for other work depends on the scheduler admitting and
  enforcing the reservation. Task scope alone is not a local GPU scheduler,
  and masking devices is not a release.
- An interrupted assignment is not resumable by itself. Partial-progress
  checkpoints remain application-owned; optimizer and random-number state are
  not preserved automatically.
- Worker metrics are sent directly to the Client Job while the task runs.

**Configuration and limits.**

- The PyTorch FedAvg, FedProx and FedOpt, NumPy FedAvg, and scikit-learn
  FedAvg, K-means and SVM recipes accept an ``execution_lifetime`` parameter
  to select job or task lifetime. FedCE stays at job lifetime, and the
  shipped scikit-learn examples continue across rounds in fresh workers.
- Task lifetime on local Process and Docker backends is deprecated; use the
  Kubernetes or Slurm task launchers. Job lifetime remains supported on
  Process, Docker, Kubernetes, and Slurm, and a federated job can mix
  job-lifetime and task-lifetime sites through per-site client apps.
- Supported executors are the Client API executor in ``in_process`` mode, the
  statistics executor, and the NPTrainer compatibility path. Client API
  ``external_process`` and attach modes, both SCAFFOLD implementations, and
  CrossSiteModelEval local-model submission are rejected in task mode, as is
  Lightning ``restore_state=True``.
- Checkpoint continuity across tasks is opt-in through
  ``flare.system_info()["task_checkpoint"]``.

Reliability
===========

- **Confirmed shutdown before release.** A failed launch or an abort no longer
  releases job credentials and resource reservations while the job or its
  child processes may still be running. Parents retain the launch handle and
  keep reconciling until the backend confirms shutdown, on Process, Docker, and
  Kubernetes. Kubernetes job Pods use a settlement finalizer, so the server
  Helm Role needs Pod ``patch`` permission (see the migration guide).

Automatic Client Job Recovery
-----------------------------

Long federated jobs often outlive a scheduler allocation. A Slurm wall-time
limit or another confirmed Client Job failure can no longer end a long-running
training task that has a usable checkpoint. The Client
Parent relaunches a confirmed-dead Client Job with the same logical job,
launcher configuration, resource reservation, credentials, and persistent
workspace; application code restores its checkpoint.

- Recovery applies to leaf Client Jobs on Process, Docker, Slurm, and
  Kubernetes. Intermediate Client Jobs in hierarchical federations keep the
  previous fail-on-crash behavior.
- On Kubernetes, recovery needs the persistent jobs volume; resident Client
  Jobs use an ``emptyDir`` workspace unless ``persist_resident_workspaces`` is
  enabled.
- Process recovery is not available for an ``external_process`` trainer.
- ``max_cj_restart_failures`` (default three consecutive failed replacement
  initializations; a successful one resets the count) bounds retries.

Networking and Deployment
=========================

- **Upgrade path.** A 2.10 client can be staged before a server upgrade: it
  waits, using a transport-only probe at a configurable interval (default 60
  seconds), until it sees a compatible server.
- **Outbound proxies.** The HTTPS (WebSocket) transport honors standard proxy
  environment variables; ``wss_proxy``, ``no_proxy``, and Basic proxy
  credentials are documented.
- **Docker GPU reservation.** Docker jobs reuse the client's GPU resource
  manager, so concurrent jobs no longer receive the same GPUs. Passthrough
  clients keep their previous behavior.
- Lightweight per-peer payload byte counters in CellNet.

Developer Experience
====================

- **Quick Start and documentation.** A rewritten README and Quick Start, a
  reorganized tutorial navigation, and self-paced Chapters 1 and 2 rewritten
  around the Recipe API.
- **Example catalog.** ``nvflare examples list`` and ``nvflare examples get
  <name>`` retrieve a curated example matching the installed NVFlare version,
  without cloning the repository.
- **Recipes, Simulator, and POC.** Progress logging and run summaries across
  workflows and recipe runs, clearer Recipe script arguments and simulation
  status, a warning when export flags are not declared by the script,
  per-deployment POC workspace isolation, and an optional sampling seed for
  client selection.

Bug Fixes
=========

Federated Statistics
--------------------

- Hierarchical variance and standard deviation below the top level are now
  computed correctly from the site standard deviations.
- Sites withheld by a privacy filter are handled in hierarchical statistics
  and excluded from the global count.
- Feature data types reported by earlier clients are preserved.
- Quantiles are preserved when a client sends an empty digest.
- Python scalars in dataframe statistics are handled.

Aggregation and Workflows
-------------------------

- PyTorch FedOpt final-global evaluation uses the correct framework, persistor,
  and locator metadata, and full cross-site evaluation reuses the registered
  locator.
- Failed FedAvg callbacks no longer publish partial aggregates.
- Aggregation rejects incompatible parameter shapes before weighted
  aggregation and preflights full-model weight diffs before mutation; built-in
  FedAvg continues with the shape-compatible client contributions (see the
  :ref:`migration_guide`).
- Fixed FedBuff reselection stalls and lost buffered updates, late
  ``ModelController`` results, cross-site evaluation exceeding
  ``max_parallel_actions`` per site, wildcard task-filter matching, and a
  workflow property conflict when assigning server tasks.
- Lightning ``max_steps`` rounds keep a constant length, and evaluation tasks
  keep the training budget.

Networking and Infrastructure
-----------------------------

- Relay authentication refreshes after a server key replacement.
- Slurm multi-node GPU step allocation.
- Transport startup and shutdown lifecycle handling, and async TCP scheme
  handling in client startup URLs.
- Site communication settings reload after workspace bootstrap.

Recipes, Simulator, and POC
---------------------------

- Simulator launch works when the interpreter or workspace path contains a
  space.
- POC readiness checks no longer fail prematurely or wait without a bound.
- Configuration format detection for multi-dot file names, and saving a
  configuration with a YAML extension.

Logging and Job Isolation
-------------------------

- Job-scoped live logs are preserved during graceful shutdown.
- Concurrent jobs no longer share server-parent context IDs and metric timers.

Examples and Research
=====================

- Federated BioNeMo Evo2 fine-tuning, Gen3 data access for study workloads,
  Nemotron 3.5 Lightning support in the NeMo PEFT example, and a Hello PyTorch
  quickstart with environment continuity.
- Collab API (technical preview): simple asynchronous MNIST and full FedBuff
  on CIFAR-10, a reusable buffered asynchronous ``AsyncRecipe`` with
  independent active-client, buffer, and refill settings, a reproducible
  asynchronous CIFAR-10 example, and a minimal distributed SQL example.
- PyTorch distillation helpers under ``nvflare.app_opt.pt.distillation``:
  supplied-data knowledge distillation and optional data-free synthesis, usable
  without a federated runtime.
- Research: AgenticFL (an agent-negotiated workflow on a 39-site retinal
  cohort), FedSCS aggregation, FedRevive, FedCoRe (Qwen3-VL missing
  modality), BreastG-FCL, and a cross-round client-drift monitoring example.

Upgrade and Compatibility
=========================

For step-by-step upgrade instructions and the actions needed for each change
below, see the :ref:`migration_guide`.

.. warning::

   2.10 sites cannot connect to 2.9 sites, whichever 2.10 features you use.
   Every 2.10 connection handshake carries a transport protocol version, and a
   peer with a missing or different version (including every 2.9 peer) is
   rejected before it attaches. Upgrade the whole federation together; see the
   migration guide for how to stage clients.

- Secure-mode projects must be re-provisioned for per-job certificates, which
  are required in secure mode (a server kit without a job CA cannot deploy
  jobs). Single sign-on and certificate-derived study memberships are
  optional.
- ``nvflare job clone``, the Admin CLI ``clone_job``, and
  ``Session.clone_job()`` are deprecated. Cloned jobs retain the original
  signatures, signer, and certificate expiration.
- Collab API (technical preview): the ``async_aggregation`` example is renamed
  ``response_callbacks``.
- Internal client terminal-report names are renamed from job failure to job
  outcome; the previous constants, class, and wire value remain as aliases.
- Process jobs now apply ``local/study_runtime.yaml`` (previously ignored).
  Review existing files before upgrading.
- Task-lifetime PyTorch jobs: unfiltered Client Job input hooks receive tensor
  references (see Memory and Large Models).

