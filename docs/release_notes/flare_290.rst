:orphan:

**************************
What's New in FLARE v2.9.0
**************************

Highlights:

- **Agent Skills** — agent-assisted federated development
- **Collaboration API** — a Python-first API for research workflows
- **Slurm job launcher** — a new HPC execution target alongside process,
  Docker, and Kubernetes
- **Large-model training** — a hardened model-transfer streaming transport
  and FedAvg validated to 72 billion parameters
- **Security hardening** — authenticated CellNet messages, internal mTLS
  by default, and hardened admin and job-signing paths

Kubernetes/OpenShift deployment and framework/recipe additions also shipped
this release; see `Also in This Release`_ below.

Agent Skills
============

FLARE Agent Skills fall into two categories:

**Conversion skills** generate a reviewable federated job from an existing
project or dataset:

- Training conversion — PyTorch, PyTorch Lightning, and Hugging Face Trainer
  are currently supported — identifies the owning framework, preserves the
  training and evaluation semantics, generates the supported Client API or
  recipe integration, validates the generated artifact, and reports
  evidence.
- Federated statistics (tabular and image) generates a ``FedStatsRecipe``
  job directly from the dataset and feature names, with no user statistics
  code required.

**Auto-FL research assistance** runs a coding-agent-directed campaign over an
existing job:

- NVFLARE owns the deterministic campaign import, execution, policy
  boundaries, and provenance.
- The coding agent proposes hypothesis-driven candidates, constrained to the
  job's fixed training budget and mutation-schema bounds.
- Its initial importer similarly supports statically recognizable NVFLARE
  Recipe and ``*Job`` patterns.

Bundled skills are validated by pre-merge security scans, including
prompt-injection and untrusted-input eval coverage, and include explicit
safeguards for site-local data and preprocessing. Install and invoke them through a coding agent as
described in :doc:`/user_guide/agent_skills/index` (see :ref:`autofl_skill`
for the Auto-FL workflow); start with the
:github_nvflare_link:`runnable Agent Skills examples
<examples/hello-world/agent-skills>` to try the conversion and
federated-statistics workflows.

Agent Skills are developer tooling, not a runtime FL API — review a
generated job before running it.

Collaboration API
==================

.. admonition:: Technical Preview

   The Collaboration API is a **technical preview** designed for researchers
   to run quick experiments. It can run and deploy on a real multi-machine
   setup, but is not recommended for production at this rollout.

The Collaboration API provides a Python-first way to express custom federated
algorithms: decorate the functions that a server or client publishes, write
the coordination logic in ordinary Python, and use ``CollabRecipe`` to
package, export, simulate, or submit the result. This suits research
workflows that don't fit a standard controller pattern. Every Collab call is
now authorized against the caller's authenticated CellNet origin before
dispatch, rejecting a caller, method, or target that doesn't match the call
envelope.

Trimmed from the ``hello-collab`` example — the client publishes an ordinary
function, and the server calls it on every client as if it were local, with
no ``Shareable``, ``DXO``, or ``FLModel`` transport objects:

.. code-block:: python

   import numpy as np

   from nvflare.collab import CollabRecipe, collab
   from nvflare.recipe import SimEnv

   INITIAL_MODEL = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

   @collab.publish  # publishes this function to clients under the name "train"
   def train(model, update_type):
       updated_model = model + 1
       return updated_model, float(updated_model.mean())

   @collab.main
   def run():
       model = INITIAL_MODEL.copy()
       for _ in range(collab.get_app_prop("num_rounds", 3)):
           # "train" here calls the published train() function on every client
           client_results = collab.clients.train(model, "full")
           updates = [update for _, (update, _) in client_results]
           model = np.mean(updates, axis=0)  # average into the next round's model
       return model

   # CollabRecipe discovers the decorated functions above for both sides.
   CollabRecipe(job_name="hello_numpy_collab").execute(SimEnv(num_clients=2))

New examples:

- :github_nvflare_link:`Hello Collab <examples/hello-world/hello-collab>` —
  a minimal FedAvg workflow
- :github_nvflare_link:`pt_async_cifar10
  <examples/advanced/collab/pt_async_cifar10>` — in-time aggregation on
  CIFAR-10 at scale (up to 1,000 logical clients)
- :github_nvflare_link:`advanced Collab examples <examples/advanced/collab>`
  — split learning, swarm learning, and in-time aggregation

Slurm Job Launcher
===================

FLARE 2.9.0 adds a new Slurm job launcher for HPC environments, joining the
existing process, Docker, and Kubernetes launchers. A long-lived NVFLARE
parent submits each client or server job process as a Slurm batch job;
Slurm selects resources while FLARE manages the federated job lifecycle.
The Slurm launcher supports:

- Apptainer, Pyxis/Enroot, and bare-Python execution backends
- GPU-aware worker setup and multi-node applications
- a shared-file worker channel for clusters where compute nodes cannot open a
  direct connection to the parent

Follow the :ref:`slurm_job_launcher` deployment guide for prerequisites,
backend setup, site configuration, and validation steps.

Large-Model Training
=====================

FLARE 2.9.0 strengthens the streaming transport used for large model
transfers, across three areas:

**Reliable Streaming** — a transfer survives interruptions instead of
failing outright:

- Unacknowledged chunks retry within bounded retry budgets, and
  receiver-confirmed completion holds a payload until the receiver has
  actually consumed it.
- A progress-aware liveness policy keeps a task download or result upload
  alive as long as bytes keep advancing, instead of failing or resending on
  a fixed wall-clock timeout; a transfer that truly stalls still fails after
  the configured idle limit. This now extends to Swarm Learning, where the
  aggregation client's result-upload progress is tracked by its exact FQCN in
  relay and hierarchical topologies instead of falling back to
  single-receiver progress tracking.
- External-trainer task materialization no longer trips a heartbeat expiry
  while a ``TASK_READY`` exchange is pending (an optional task-wait timeout
  still bounds it).

**Throughput and flow control** — sender and receiver stay in sync under
load:

- The sender now tells the receiver its effective chunk, window, ACK, and
  retry-pending limits per stream, so mismatched endpoint settings can't
  stall flow control.
- Receiver reassembly capacity tracks the negotiated stream window instead
  of a fixed chunk count, so scheduler-induced chunk reordering doesn't
  abort healthy transfers under load.
- Pipelined tensor downloads, prefetching, and ``TCP_NODELAY`` improve
  throughput; oversized blobs fail before transmission with an actionable
  error, and a failed streamed result send now retries instead of silently
  dropping the result.
- Active administrative result downloads refresh their bound HCI session as
  bytes advance, so a healthy long download doesn't expire as idle.

**Memory** — peak server RSS stays flatter as models and client counts grow:

- Tensor disk offload during aggregation, previously FedAvg-only, now also
  covers Scaffold, FedOpt, and Swarm.
- Pass-through tensor broadcasts release their source transaction as soon as
  downstream consumers finish, instead of retaining a model-sized object per
  aggregation round.
- The default maximum streamed blob size is 4 GiB, and remains configurable.

With suitable infrastructure and configuration, FedAvg has been validated
for federated LLM training at scales up to 72 billion parameters. See
:ref:`notes_on_large_models` for deployment sizing and large-model
operational guidance.

.. list-table::
   :widths: 55 45

   * - .. image:: ../resources/flare_290_72b_training_elapsed_time.png
          :width: 460px
          :height: 307px

       Elapsed time, 1.7B-72B (measured, 2 training rounds)
     - .. image:: ../resources/flare_290_72b_training_server_memory.png
          :width: 390px
          :height: 260px

       FedAvg server peak memory, 1.7B-72B (measured, 2 training rounds)

Each configuration in both charts ran for 2 FedAvg rounds; the measurements
characterize per-round transfer time and server peak memory, not a full
convergence training run.

With pass-through download on, Swarm Learning's tensor disk offload lowers
the fixed aggregator's peak memory, and the savings widen with model size;
non-aggregator sites stay approximately flat at every size, since disk
offload targets contribution handling at the aggregator rather than the
per-site learner footprint (external-process, fixed aggregator, 4 clients,
30 rounds):

.. image:: ../resources/flare_290_swarm_disk_offload_memory.png
   :width: 850px
   :align: center

Aggregator (site-1) peak container memory, disk offload OFF vs. ON:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model
     - OFF peak
     - ON peak
     - Reduction
   * - 5 GB synthetic
     - 48.49 GiB
     - 37.08 GiB
     - 23.5%
   * - 30 GB (Qwen2.5-14B)
     - 236.80 GiB
     - 150.80 GiB
     - 36.3%
   * - 60 GB synthetic
     - 452.20 GiB
     - 287.40 GiB
     - 36.4%

Non-aggregator sites (site-2/3/4) moved by -1.9% to +3.0% across all three
model sizes -- run-to-run peak variation, not a disk-offload effect.

Security Hardening
====================

FLARE 2.9.0 hardens the internal transport and admin access, on top of
moving job-process bootstrap credentials off the command line (see
Compatibility and Migration Notes below for that migration's
requirements):

- **CellNet message authentication.** Cell payload encryption moves from
  unauthenticated AES-CBC to signed AES-256-GCM envelopes, and the sender
  signature on every message — including cached-key paths — is now
  verified before it's trusted, closing a ciphertext bit-flipping
  exposure. This is a wire-format change: a 2.9 peer rejects the legacy
  unversioned ciphertext, so encrypted CellNet participants must upgrade
  to 2.9 together.

- **Internal mTLS by default.** Internal CellNet TCP links between a
  parent and its job processes now default to mutual TLS across Docker,
  Slurm, Kubernetes, and Network Attach deployments, each with an
  explicit clear-transport opt-out for sites that intentionally run
  without it.

- **Certless admin session and listener hardening.** An admin session
  token that can't be signature-verified now fails closed instead of
  falling back; admin, TCP, and SimEnv listeners bind to explicit
  loopback or configured hosts instead of a broad wildcard default.

- **Cross-client authentication now routes through the server.** A
  message between different client families is authenticated through the
  server trust boundary even when a direct or cached peer endpoint would
  otherwise be used.

- **New ``require_signed_jobs`` policy.** A client-local policy rejects
  unsigned job deployment bytes at the receiving site; exact-byte
  signature verification is preserved even when unsigned jobs are
  otherwise allowed.

- **CLI, diagnostics, and Recipe secret handling.** CLI and runtime
  diagnostics redact sensitive values more consistently, and Recipe APIs
  add safeguards for declaring and handling secrets; see
  :ref:`recipe_secrets`.

Also in This Release
======================

- **Kubernetes and OpenShift deployment** — stage a prepared kit as
  Kubernetes ConfigMaps and Secrets and mount them through the generated Helm
  chart; the workspace PVC stays mounted for writable runtime state.

  - Run ``nvflare deploy k8s stage`` after ``nvflare deploy prepare``; add
    ``--kubectl oc`` for OpenShift and run ``nvflare deploy k8s unstage``
    after Helm uninstall.
  - See :ref:`helm_chart` and the :github_nvflare_link:`OpenShift
    <examples/devops/openshift>` / :github_nvflare_link:`multicloud
    <examples/devops/multicloud>` examples.

- **Hugging Face Client API** — federate an existing ``Trainer`` or TRL
  ``SFTTrainer`` through ``flare.patch(trainer)``.

  - FLARE owns round exchange, global-weight loading, local-budget
    enforcement, rank-0 communication, checkpoint continuity, and metric
    reporting.
  - See :ref:`hf_client_api` and the :github_nvflare_link:`Hello Hugging Face
    example <examples/hello-world/hello-huggingface>`.

- **Client API Attach and Recipe updates** — Attach mode lets a long-lived,
  application-owned external trainer connect without starting a new process
  each round.

  - See :ref:`client_api_attach` and the :github_nvflare_link:`example
    <examples/advanced/client-api-attach>`.
  - Recipe updates add the concrete PyTorch FedBPT entry point, expose
    ``key_metric_mode`` for FedAvg recipes, and improve PyTorch workflow
    support for FedProx, SCAFFOLD, Swarm, and model-selection behavior.

Compatibility and Migration Notes
=================================

- **HUB support removed.** The deprecated FL HUB feature and its
  remaining runtime, documentation, and test surface are removed,
  including the ``nvflare.app_common.hub`` implementation and its
  component registry entries.

- **Internal CellNet TCP links default to mTLS.** Docker, Slurm,
  Kubernetes, and Network Attach deployments now default to mutual TLS
  between a parent and its job processes; a site that intentionally runs
  without it needs an explicit clear-transport opt-out. See
  `Security Hardening`_ above for the full list of hardening changes.

  - Kubernetes and Slurm sites use the participant certificate in both TLS
    roles, which requires a certificate allowing both ``clientAuth`` and
    ``serverAuth``. A startup kit with a role-restricted certificate — for
    example, from NVFlare 2.8 distributed provisioning, which issues only
    ``clientAuth`` or only ``serverAuth`` — must be re-provisioned before
    using the Kubernetes or Slurm launcher; unrestricted (no-EKU)
    certificates remain compatible. After re-provisioning, Kubernetes
    sites rerun ``nvflare deploy prepare`` and Slurm sites rebuild the
    runtime workspace.

- **``poc start``/``poc stop`` preserve every repeated flag.** Earlier
  versions silently kept only the last ``-p``/``--service`` or
  ``-ex``/``--exclude`` value. ``poc stop`` now also honors participant
  exclusions, and now waits for targeted and exclusion-based shutdowns to
  complete before returning ``status: stopped`` (use ``--no-wait`` for
  fire-and-forget). A bare ``poc start`` continues to start the server and
  clients without an admin console.

- **Docker job-controlled launcher options are now restricted.** Jobs may
  control only ``image``, ``python_path``, ``entrypoint``, ``num_of_gpus``,
  and ``shm_size`` through their launcher metadata; selecting ``image``,
  ``python_path``, or ``entrypoint`` requires BYOC authorization at each
  receiving site.

  - Previously job-controlled Docker SDK options such as ``ipc_mode`` and
    ``device_requests`` are now site-owned, configured through
    ``default_job_container_kwargs`` or a study's ``docker_kwargs``;
    launcher-owned options such as mounts and networks remain fixed.
  - New jobs with unsupported options are rejected at submission; jobs
    stored before an upgrade are checked again and can fail at launch
    until their metadata is migrated.

- **Portable job resource fields are now reserved.** The flat
  ``resource_spec`` names ``num_of_gpus``, ``num_of_cpus``, and ``memory``
  must use the documented portable types. Custom resource managers that
  previously interpreted these names differently must migrate to the
  portable types or rename their custom fields. Legacy nested resource
  specifications without ``@default`` remain unchanged.

- **Collab calls carry a versioned authorization envelope.** Calls are
  accepted only from authenticated participants in the same job; all
  sites running a Collab job must use NVFlare 2.9 or newer.

  - A 2.8-or-older Collab sender lacks this envelope and its calls are
    rejected: a 2.9 sender receives an immediate ``COMM_ERROR``, while a
    2.8-or-older sender can instead observe a request timeout, since it
    lacks the dedicated stream-error correlation used by 2.9. The
    receiving site logs that the peer may be running an older NVFlare
    version to make this mixed-version failure diagnosable.

- **Streaming transport defaults are larger.** The sender's default
  streaming window / ACK interval move from 2.8's 16 MiB / 4 MiB to 64 MiB
  / 16 MiB (``STREAM_WINDOW_SIZE`` / ``STREAM_ACK_INTERVAL``, documented in
  ``dev_tools/f3/comm_config.yml``); ``TCP_NODELAY`` is now on by default,
  reducing request/ACK latency.

- **Job-process bootstrap credentials move off the command line.**
  Launchers deliver them through the job process environment instead (a
  per-job Kubernetes Secret via ``env[].valueFrom.secretKeyRef``).

  - No fallback: Docker/Kubernetes job images must run NVFlare 2.9 or
    newer, or they fail immediately at argument parsing when launched by
    a 2.9 CP/SP. The CLI path is retained, so an older parent launching a
    newer job image is unaffected.
  - A custom launcher that renders worker commands from
    ``generate_client_command`` / ``generate_server_command`` and
    implements ``launch_job`` directly must also export
    ``get_credential_env(job_args)`` into the child environment.
  - Launcher Kubernetes RBAC now needs the ``patch`` and ``delete`` verbs
    on Secrets (already in the generated Helm role templates).

- **Patched Lightning clients report real per-round steps.**
  ``NUM_STEPS_CURRENT_ROUND`` is now the actual per-round change in
  ``trainer.global_step`` instead of ``trainer.estimated_stepping_batches``,
  correcting cumulative aggregation over-weighting in later rounds when
  ``update_fit_loop=True``.

  - ``global_step`` counts steps across all optimizers, so a
    multi-optimizer FedAvg client reports their combined step count
    unless it supplies ``NUM_STEPS_CURRENT_ROUND`` explicitly; explicit
    client metadata is still preserved.

- **Patched Lightning clients transmit pre-fit validation metrics.** A
  metric captured by an explicit ``trainer.validate()`` call before
  ``trainer.fit()`` is now transmitted regardless of
  ``train_with_evaluation``; sanity-check and in-fit validation metrics
  are still not transmitted as global-model scores. This enables
  ``IntimeModelSelector`` and best-global-model persistence for
  recipe-based Lightning jobs.

  - ``train_with_evaluation=True`` still requires validation metrics;
    otherwise metrics remain optional, but ``False`` no longer suppresses
    metrics from an explicit pre-fit validation.
  - An application that must keep such metrics local should omit the
    explicit pre-fit validation; if it still requires one locally, a
    custom task-result filter must remove ``MetaKey.INITIAL_METRICS``
    before the result reaches the server.

- **``SimpleIntimeModelSelector`` (CCWF) now handles dict-valued metrics.**
  It selects a scalar ``key_metric`` (default ``val_accuracy``) from
  dict-valued ``INITIAL_METRICS`` payloads instead of failing with a
  logged ``TypeError`` that silently disabled best-model selection.

  - Swarm jobs whose metric dicts contain the configured key change from
    selection-inert to active best-global-model tracking without a config
    change; dict payloads lacking the key, and non-numeric values, are
    skipped with a warning, so configure ``key_metric`` to match the
    reported metric name.
  - A new ``negate_key_metric`` argument supports lower-is-better metrics
    such as losses.

- **FedProx recipe rename.** Recipe discovery now exposes the concrete
  PyTorch ``FedProxRecipe`` as ``fedprox-pt`` and no longer advertises the
  ``fedprox-tf`` manual pattern as a concrete recipe. TensorFlow clients
  can still combine a FedAvg recipe with ``TFFedProxLoss`` explicitly.

- **Unified Client API execution paths.** ``ClientAPIExecutor``
  consolidates NVFlare's trainer-process ownership patterns behind one
  Client API executor; jobs generated with FLARE 2.9 require a client
  runtime that provides it and are not runnable on older client runtimes.

  - ``in_process`` replaces the previous ``InProcessClientAPIExecutor``.
  - ``external_process`` replaces the former ``ClientAPILauncherExecutor``
    stack (``LauncherExecutor``, ``SubprocessLauncher``,
    ``TaskExchanger``, ``FlareAgent``, ``BaseScriptRunner``,
    ``ExternalConfigurator``, and the ``Pipe``/``PipeHandler``
    implementations including ``FilePipe`` and ``CellPipe``) for trainers
    launched and owned by NVFlare: the launched trainer creates its own
    Cell, with a prescribed FQCN from a typed, owner-only bootstrap file,
    and connects to the client job's Cell over an authenticated,
    liveness-checked session.
  - ``attach`` provides a standard Client API migration path for the
    independently managed trainer pattern served by ``IPCExchanger`` and
    ``IPCAgent``. It preserves the server-facing trust boundary and
    CP-routed topology of that path while adding an explicit session
    protocol, Attach profiles, and ``flare.receive()``/``flare.send()``
    integration.
  - Custom parameter transformations (formerly ``ParamsConverter`` and
    the framework-specific converter components) belong in trainer code
    around ``flare.receive()``/``flare.send()``; common functions remain
    available in ``nvflare.client.converter_utils``.
  - Recipe-level ``pipe_type`` and ``pipe_root_path`` options are also
    removed; transport is selected through site communication
    configuration. The F3 ``FileDriver`` remains available as scheme
    ``shared-file`` for an attached trainer; a launched external-process
    trainer instead requires a clear TCP listener bound to loopback.
  - ``ScriptRunner`` selects ``ClientAPIExecutor(in_process)`` by
    default, or ``ClientAPIExecutor(external_process)`` when
    ``launch_external_process=True``, and no longer performs a build-time
    PyTorch or TensorFlow import check. A client app may contain only one
    ``ClientAPIExecutor``; configurations that previously added multiple
    script runners to one site must combine the scripts behind one entry
    point and dispatch on the Client API task name.

- **Swarm can combine tensor streaming with disk offload.** Set
  ``aggregation_format=ExchangeFormat.PYTORCH`` and
  ``enable_tensor_disk_offload=True`` on ``SwarmLearningRecipe``; the same
  offload flag is available on ``SwarmClientConfig`` for Job API users.

- **``SwarmLearningRecipe`` now configures best-model selection by
  default.** Use ``key_metric`` to select a dictionary-valued validation
  metric and ``key_metric_mode="min"`` for lower-is-better metrics.

  - Clients must report a pre-training validation metric with the
    configured name for selection to occur; jobs without that metric
    continue to persist the last global model but do not create
    ``best_FL_global_model.pt``. Set ``key_metric=None`` to opt out and
    preserve the pre-2.9 last-model-only behavior.
  - Selection skips round 0, so a one-round job does not create a
    best-model checkpoint. With ``key_metric_mode="min"``, Swarm
    best-metric logs and records expose the negated comparison value (for
    example, a loss of 2.31 is shown as -2.31).
  - ``client_config_overrides`` can no longer replace ``model_selector``:
    migrate the former ``{"model_selector": None}`` opt-out to
    ``key_metric=None``, and use ``BaseSwarmLearningRecipe`` with an
    explicit ``SwarmClientConfig`` for a custom selector.

- **Auto-FL campaigns now honor the job's native metric direction**
  (``key_metric_mode`` or a matching same-metric ``stop_cond``) instead of
  assuming maximization, so raw lower-is-better objectives no longer need
  to be negated. Campaign admission also fails closed in the following
  cases:

  - An obvious lower-is-better metric such as ``val_loss`` that relies
    only on NVFlare's implicit ``max`` default is rejected until the job
    declares ``key_metric_mode="min"``.
  - A job passing a custom ``model_selector`` is rejected, because that
    component supersedes ``key_metric_mode`` and its selection direction
    can't be imported deterministically. Remove the custom selector and
    expose its criterion as a declared ``key_metric`` with
    ``key_metric_mode`` before initializing a campaign.
  - A requested metric that differs from the job's key metric is rejected
    unless ``mutation_schema.yaml`` declares the requested and
    optimization metric bridge.
  - Job constructor calls that pass positional arguments, ``*args``, or
    ``**kwargs`` also fail closed, since dynamic arguments could hide the
    metric, direction, or fixed training budget; rewrite the call with
    keyword-only arguments and no splats.
  - ``SimEnv`` calls must pin a positive explicit ``num_clients``, or
    expose a non-empty static ``clients`` list whose length is pinned.
  - Experimental legacy minimization campaigns without direction
    provenance must be re-initialized in a fresh workspace.

- **External-process Client API: an accepted lazy result can't be
  withdrawn.** Losing a trainer after its lazy result envelope has been
  accepted now fails the run as ``EXECUTION_EXCEPTION``, even if a
  controller's ``min_responses`` threshold could otherwise tolerate a
  missing client — the accepted envelope may already have exposed
  references to downstream consumers. An explicit job abort that wins the
  terminal-state race remains ``ABORTED``.
