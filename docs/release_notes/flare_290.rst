:orphan:

**************************
What's New in FLARE v2.9.0
**************************

NVIDIA FLARE 2.9.0 focuses on four major capabilities: agent-assisted
federated development, research-oriented collaborative workflows,
Slurm-managed HPC execution, and reliable large-model training. The release
also includes operational, security, framework, and Recipe API improvements.

Main Features
=============

- **Agent Skills**: installable NVFLARE-owned skills give coding agents guided,
  reviewable workflows for converting PyTorch, PyTorch Lightning, and Hugging
  Face Trainer projects to federated jobs; producing federated statistics; and
  diagnosing generated jobs. The skills include source inspection, validation,
  and data-locality guardrails.
- **Collaboration API**: define collaborative workflows as ordinary Python
  functions or classes and package them with ``CollabRecipe``. The API removes
  much of the controller, executor, and payload boilerplate required by custom
  workflows while preserving explicit server/client behavior. New examples
  cover synchronous FedAvg, asynchronous PyTorch training, and split learning.
- **Slurm enhancement for HPC**: run NVFLARE client and server job processes as
  Slurm allocations, with scheduler-managed submission, monitoring, and
  cancellation. Sites can use Apptainer, Pyxis/Enroot, or trusted bare-Python
  execution, and can use the shared-file worker channel where compute nodes
  cannot connect directly to the parent.
- **Large-model support**: reliable F3 streaming adds chunk retries,
  receiver-confirmed completion, progress-aware liveness, and negotiated
  flow-control settings for long-running model transfers. Tensor disk offload
  is available for PyTorch Swarm aggregation as well as FedAvg, and FedAvg has
  been validated for federated LLM training up to 72 billion parameters with
  suitable infrastructure and configuration.

Additional Platform Improvements
================================

- **Kubernetes and OpenShift deployment**: stage prepared kit configuration as
  Kubernetes ConfigMaps and Secrets, then mount those resources through the
  generated Helm chart while preserving the workspace PVC for writable runtime
  state. The same flow is available to Kubernetes, OpenShift, and multicloud
  deployment examples.
- **Security and credential handling**: job-process bootstrap credentials are
  delivered through the process environment, including per-job Kubernetes
  Secrets, instead of process command lines. CLI and runtime diagnostics more
  consistently redact sensitive values, and Recipe APIs provide safeguards for
  declaring and handling secrets.
- **Framework and Recipe APIs**: federate Hugging Face ``Trainer`` and TRL
  ``SFTTrainer`` scripts through the Hugging Face Client API; keep a long-lived
  external trainer connected through Client API Attach mode; and use the new
  concrete PyTorch FedBPT recipe entry point.

Main Feature Details
====================

Collaboration API
-----------------

The Collaboration API provides a Python-first way to express custom federated
algorithms. Decorate the functions that a server or client publishes, write the
coordination logic in ordinary Python, and use ``CollabRecipe`` to package,
export, simulate, or submit the result. This is especially useful for research
workflows that do not fit a standard controller pattern, including asynchronous
aggregation and split learning. See the ``hello-collab`` and advanced
collaboration examples for runnable starting points:
:github_nvflare_link:`Hello Collab <examples/hello-world/hello-collab>` for a
minimal FedAvg workflow, and :github_nvflare_link:`advanced Collab examples
<examples/advanced/collab>` for asynchronous aggregation and split learning.

Agent Skills
------------

FLARE Agent Skills help coding agents turn an existing training project into a
reviewable federated job. They identify the owning framework, preserve the
training and evaluation semantics, generate the supported Client API or recipe
integration, validate the generated artifact, and report evidence. The skills
also cover federated statistics, Auto-FL research assistance, and job
diagnostics, with explicit safeguards for site-local data and preprocessing.
Install and invoke the skills through a coding agent as described in
:doc:`/user_guide/agent_skills/index`; see :ref:`autofl_skill` for the
Auto-FL workflow. To try the conversion and federated-statistics workflows,
start with the :github_nvflare_link:`runnable Agent Skills examples
<examples/hello-world/agent-skills>`.

Agent Skills are developer tooling, not a runtime FL API or a substitute for
code review and job validation. The conversion skills currently target standard
PyTorch, PyTorch Lightning, and Hugging Face Trainer workflows. They preserve
explicitly supported Recipe and Client API paths, and ask or stop rather than
guess when framework ownership, source semantics, required data handling, or
runtime configuration is ambiguous. Auto-FL's initial importer similarly
supports statically recognizable NVFLARE Recipe and ``*Job`` patterns.

Security and Credential Handling
--------------------------------

Job-process bootstrap credentials are no longer passed on process command
lines. Launchers deliver them through the child-process environment, and
Kubernetes uses a per-job Secret. CLI and runtime diagnostics more consistently
redact authentication tokens and other sensitive values. Recipe APIs also
provide safeguards for declaring and handling secrets; see :ref:`recipe_secrets`
for the Recipe secret contract.

HPC and Large-Model Training
============================

Slurm Job Launcher
------------------

FLARE 2.9.0 adds Slurm-native job execution for HPC environments. A long-lived
NVFLARE parent submits each client or server job process as a Slurm batch job;
Slurm selects resources while FLARE manages the federated job lifecycle. The
launcher supports Apptainer, Pyxis/Enroot, and bare-Python backends, GPU-aware
worker setup, multi-node applications, and a shared-file worker channel for
clusters where compute nodes cannot open a direct connection to the parent.
Follow the :ref:`slurm_job_launcher` deployment guide for prerequisites,
backend setup, site configuration, and validation steps.

Large-Model Transport, Reliability, and Memory
----------------------------------------------

FLARE 2.9.0 strengthens the F3 transport for long-running model transfers.
When reliable streaming is enabled, unacknowledged chunks are retried within
bounded retry budgets. Receiver-confirmed completion lets the sender retain a
payload until the intended receiver has consumed it. A progress-aware liveness
policy keeps task downloads and result uploads alive while bytes continue to
advance, instead of failing or resending solely because a fixed wall-clock
timeout elapsed; transfers that stop making progress still fail after the
configured idle limit. External trainer task materialization no longer trips a
heartbeat expiry while a ``TASK_READY`` exchange is pending; an optional task
wait timeout continues to provide an absolute bound.

The transport now communicates the sender's effective chunk, window, ACK, and
retry-pending limits to the receiver for each stream, preventing incompatible
endpoint settings from stalling flow control. Receiver reassembly capacity is
derived from the negotiated stream window rather than a fixed chunk count, so
scheduler-induced chunk reordering does not abort healthy transfers under load.
Pipelined tensor downloads, prefetching, and ``TCP_NODELAY`` improve transfer
throughput. Oversized blobs fail before transmission with an actionable error,
and task-result delivery retries a failed streamed send rather than silently
dropping the result. Active administrative result downloads also refresh their
bound HCI session as bytes advance, preventing a healthy long download from
expiring as idle.

PyTorch Swarm can use tensor disk offload during aggregation, reducing peak
memory pressure by materializing incoming tensors through temporary disk
storage. Pass-through tensor broadcasts now release their source transactions
after downstream consumers complete, preventing model-sized retained objects
from accumulating across aggregation rounds. The default maximum streamed blob
size is 4 GiB, while remaining finite and configurable. With suitable
infrastructure and configuration, FedAvg has been validated for federated LLM
training at scales up to 72 billion parameters. See
:ref:`notes_on_large_models` for deployment sizing and large-model operational
guidance.

Kubernetes and OpenShift Deployment
-----------------------------------

Use ``nvflare deploy k8s stage`` after ``nvflare deploy prepare`` to create a
ConfigMap from the prepared ``local/`` directory and a Secret from
``startup/``. The command patches the generated ``helm_chart/values.yaml`` so
the parent pod mounts both resources at the expected workspace paths. The
workspace PVC remains mounted for writable runtime state, including jobs,
snapshots, logs, and server transfer storage; it no longer needs to transport
the startup-kit files.

Set ``--namespace``, ``--local-configmap``, and ``--startup-secret`` when the
default staged-resource names or target namespace do not match site policy. Use
``--kubectl oc`` when staging into OpenShift with ``oc``. After Helm uninstall,
run ``nvflare deploy k8s unstage`` to remove the staged resources and clear the
chart references. See :ref:`helm_chart`, the
:github_nvflare_link:`OpenShift example <examples/devops/openshift>`, and the
:github_nvflare_link:`multicloud Kubernetes example <examples/devops/multicloud>`.

Framework Integrations and Recipes
==================================

Hugging Face Client API
-----------------------

The Hugging Face Client API lets an existing ``Trainer`` or TRL ``SFTTrainer``
participate in federated training through ``flare.patch(trainer)``. FLARE owns
round exchange, global-weight loading, local-budget enforcement, rank-0
communication, checkpoint continuity, and metric reporting while the training
script retains its normal model, dataset, optimizer, scheduler, and callback
construction.

See :ref:`hf_client_api` for the API contract and the
:github_nvflare_link:`Hello Hugging Face example
<examples/hello-world/hello-huggingface>` for a runnable Qwen LoRA fine-tuning
job.

Client API Attach and Recipe Updates
------------------------------------

Client API Attach mode supports a long-lived, application-owned external
trainer that connects to FLARE without starting a new training process for each
round. Recipe updates add the concrete PyTorch FedBPT entry point, expose
``key_metric_mode`` for FedAvg recipes, and improve PyTorch workflow support
for FedProx, SCAFFOLD, Swarm, and model-selection behavior.
Use :ref:`client_api_attach` and the :github_nvflare_link:`Client API Attach
example <examples/advanced/client-api-attach>` when an external application,
rather than NVFLARE, owns the trainer process lifecycle.

Compatibility and Migration Notes
=================================

- Docker jobs may now control only ``image``, ``python_path``, ``entrypoint``,
  ``num_of_gpus``, and ``shm_size`` through their launcher metadata. Selecting
  ``image``, ``python_path``, or ``entrypoint`` requires BYOC authorization at
  each receiving site. Previously job-controlled Docker SDK options such as
  ``ipc_mode`` and ``device_requests`` are now site-owned and can be configured
  through ``default_job_container_kwargs`` or a study's ``docker_kwargs``;
  launcher-owned options such as mounts and networks remain fixed. New jobs
  containing unsupported options are rejected at submission; jobs stored before
  an upgrade are checked again and can fail at launch until their metadata is
  migrated.
- The flat ``resource_spec`` names ``num_of_gpus``, ``num_of_cpus``, and ``memory`` are now reserved portable fields
  and must use the documented portable types. Custom resource managers that previously interpreted these names
  differently must migrate to the portable types or rename their custom fields. Legacy nested resource specifications
  without ``@default`` remain unchanged.
- Collab calls now carry a versioned authorization envelope and are accepted
  only from authenticated participants in the same job. All sites that run a
  Collab job must use NVFlare 2.9 or newer; a 2.8 or older Collab sender lacks
  this envelope and its calls are rejected. A 2.9 sender receives an immediate
  ``COMM_ERROR``; a 2.8 or older sender can instead observe a request timeout
  because it lacks the dedicated stream-error correlation used by 2.9. The
  receiving site logs that the peer may be running an older NVFlare version to
  make this mixed-version failure diagnosable.
- F3 retains its 16 MiB streaming-window and 4 MiB ACK-interval defaults for
  compatibility and bounded per-stream memory. High-bandwidth deployments may
  opt into larger values on all endpoints; ``dev_tools/f3/comm_config.yml``
  provides a 64 MiB/16 MiB tuning example. TCP connections now enable
  ``TCP_NODELAY`` by default to reduce request/ACK latency.
- Job-process bootstrap credentials (auth token, token signature, session ID)
  are no longer passed as command-line arguments. Launchers deliver them
  through the job process environment; on Kubernetes they ride a per-job
  Secret referenced via ``env[].valueFrom.secretKeyRef``. There is no fallback
  machinery, so Docker and Kubernetes job images must run NVFlare 2.9 or
  newer: a job image pinned to 2.8 or earlier fails immediately at argument
  parsing when launched by a 2.9 CP/SP. The CLI path is retained, so an older
  parent launching a newer job image is unaffected. Custom launchers that
  render worker commands from ``generate_client_command`` /
  ``generate_server_command`` and implement ``launch_job`` directly must also
  export ``get_credential_env(job_args)`` into the child environment. Launcher
  Kubernetes RBAC now needs the ``patch`` and ``delete`` verbs on Secrets
  (included in the generated Helm role templates).
- Patched PyTorch Lightning clients now report ``NUM_STEPS_CURRENT_ROUND`` as
  the actual per-round change in ``trainer.global_step`` instead of
  ``trainer.estimated_stepping_batches``. This corrects cumulative aggregation
  over-weighting in later rounds when ``update_fit_loop=True``. Because
  ``global_step`` counts completed optimizer steps across optimizers, a
  multi-optimizer FedAvg client reports their combined step count unless it
  supplies ``NUM_STEPS_CURRENT_ROUND`` explicitly; explicit client metadata is
  still preserved.
- Patched PyTorch Lightning clients now transmit validation metrics captured by
  an explicit ``trainer.validate()`` call before ``trainer.fit()``, regardless
  of ``train_with_evaluation``. Sanity-check and in-fit validation metrics are
  not transmitted as global-model scores. Setting
  ``train_with_evaluation=True`` continues to require validation metrics;
  otherwise metrics remain optional, but ``False`` no longer suppresses metrics
  from an explicit pre-fit validation. This enables ``IntimeModelSelector`` and
  best-global-model persistence for recipe-based Lightning jobs. Applications
  that must keep such metrics local should omit the explicit pre-fit validation;
  if they still require it locally, a custom task-result filter must remove
  ``MetaKey.INITIAL_METRICS`` before the result reaches the server.
- ``SimpleIntimeModelSelector`` (CCWF) now selects a scalar ``key_metric``
  (default ``val_accuracy``) from dict-valued ``INITIAL_METRICS`` payloads
  instead of failing with a logged ``TypeError`` that silently disabled
  best-model selection. Swarm jobs whose metric dicts contain the configured
  key change from selection-inert to active best-global-model tracking without
  a config change; dict payloads lacking the key, and non-numeric values, are
  skipped with a warning, so configure ``key_metric`` to match the reported
  metric name. A new ``negate_key_metric`` argument supports lower-is-better
  metrics such as losses.
- Recipe discovery now exposes the concrete PyTorch ``FedProxRecipe`` as
  ``fedprox-pt`` and no longer advertises the ``fedprox-tf`` manual pattern as
  a concrete recipe. TensorFlow clients can continue to combine a FedAvg
  recipe with ``TFFedProxLoss`` explicitly.
- The legacy Client API execution stack has been removed. This includes
  ``ParamsConverter``, the framework-specific converter components,
  ``InProcessClientAPIExecutor``, ``ClientAPILauncherExecutor``,
  ``LauncherExecutor``, ``SubprocessLauncher``, ``TaskExchanger``,
  ``FlareAgent``, ``BaseScriptRunner``, ``ExternalConfigurator``, and the
  ``Pipe``/``PipeHandler`` implementations (including ``FilePipe`` and
  ``CellPipe``). Use ``ClientAPIExecutor`` with
  ``in_process``, ``external_process``, or ``attach`` execution mode. Custom
  parameter transformations belong in trainer code around
  ``flare.receive()``/``flare.send()``; common functions remain available in
  ``nvflare.client.converter_utils``.
- Recipe-level ``pipe_type`` and ``pipe_root_path`` options have been removed.
  Transport is selected through site communication configuration. The F3
  ``FileDriver`` remains available as scheme ``shared-file`` for either a
  launched external process or an attached trainer.
- ``ScriptRunner`` now exports ``ClientAPIExecutor`` for both in-process and
  external-process execution. Jobs generated with FLARE 2.9 therefore require
  a client runtime that provides this executor and are not runnable on older
  client runtimes. ``ScriptRunner`` no longer performs a build-time PyTorch or
  TensorFlow import check; ensure the required framework dependencies are
  available in the execution environment. A client app may contain only one
  ``ClientAPIExecutor``; configurations that previously added multiple
  script runners to one site must combine the scripts behind one entry point
  and dispatch on the Client API task name.
- PyTorch swarm learning can now combine client-to-client tensor streaming with
  aggregation-client tensor disk offload. Set
  ``aggregation_format=ExchangeFormat.PYTORCH`` and
  ``enable_tensor_disk_offload=True`` on ``SwarmLearningRecipe``. The same
  offload flag is available on ``SwarmClientConfig`` for Job API users.
- In external-process Client API mode, losing a trainer after its lazy result
  envelope has been accepted now fails the run as ``EXECUTION_EXCEPTION`` even
  if a controller's ``min_responses`` threshold could otherwise tolerate a
  missing client. The accepted envelope may already have exposed references to
  downstream consumers and cannot be safely withdrawn. An explicit job abort
  that wins the terminal-state race remains ``ABORTED``.
- ``SwarmLearningRecipe`` now configures client-side best-model selection by
  default. Use ``key_metric`` to select a dictionary-valued validation metric
  and ``key_metric_mode="min"`` for lower-is-better metrics. Clients must report
  a pre-training validation metric with the configured name for selection to
  occur; jobs without that metric continue to persist the last global model but
  do not create ``best_FL_global_model.pt``. Set ``key_metric=None`` to opt out
  and preserve the pre-2.9 last-model-only behavior. Selection skips round 0,
  so a one-round job does not create a best-model checkpoint. With
  ``key_metric_mode="min"``, Swarm best-metric logs and records expose the
  negated comparison value (for example, a loss of 2.31 is shown as -2.31).
  ``client_config_overrides`` can no longer replace ``model_selector``: migrate
  the former ``{"model_selector": None}`` opt-out to ``key_metric=None``, and
  use ``BaseSwarmLearningRecipe`` with an explicit ``SwarmClientConfig`` for a
  custom selector.
