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
Auto-FL workflow.

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
- Recipe discovery now exposes the concrete PyTorch ``FedProxRecipe`` as
  ``fedprox-pt`` and no longer advertises the ``fedprox-tf`` manual pattern as
  a concrete recipe. TensorFlow clients can continue to combine a FedAvg
  recipe with ``TFFedProxLoss`` explicitly.
- CellPipe cell names now keep the runtime token and pipe mode in one
  explicitly marked, ``~``-delimited FQCN leaf segment
  (``site-1.cellpipe~plain~<job-id>~active``, or
  ``<relay>.cellpipe~alias~<site>~<job-id>~active`` behind a relay) so a
  pipe cell's FQCN parent matches the cell it actually connects to and pipe
  names can never be confused with other cell names. As part of this change,
  CellPipe validates tokens at construction: tokens must be non-empty, may
  not contain the reserved ``~`` separator, and may not contain ``.`` when
  the pipe connects to the site's own CP or a relay. Custom
  ``FlareAgentWithCellPipe`` agent ids that violate these rules now fail fast
  with a ``ValueError`` instead of producing unroutable cell names.
- Both ends of a CellPipe pair derive each other's cell names independently,
  so a Client Job process and an external training process must run the same
  NVFlare naming scheme. A training environment pinned to an older NVFlare
  fails with "peer FQCN mismatch" when paired with a 2.9 CJ; align the
  training environment's NVFlare version with the site's. Only the flat
  whole-FQCN alias used by NVFlare 2.8 and earlier (a root-connected pipe
  named ``<site>_<token>_<mode>``) is still recognized for backward
  compatibility. The forms used through 2.8 when nested under a CP or relay
  (``<parent>.<site>_<token>_<mode>``) are not, because an unmarked leaf
  inside a longer FQCN is indistinguishable from a real cell of that name.
  When upgrading to 2.9, upgrade a site and its relay together, including
  sites currently running NVFlare 2.8.
- ``ScriptRunner`` now exports ``ClientAPIExecutor`` for both in-process and
  external-process execution. Jobs generated with FLARE 2.9 therefore require
  a client runtime that provides this executor and are not runnable on older
  client runtimes. ``ScriptRunner`` no longer performs a build-time PyTorch or
  TensorFlow import check; ensure the required framework dependencies are
  available in the execution environment. Code that explicitly passes
  ``pipe_connect_type`` (including its former default value) or supplies a
  custom ``task_pipe`` must use ``BaseScriptRunner``. A client app may contain
  only one ``ClientAPIExecutor``; configurations that previously added multiple
  script runners to one site must combine the scripts behind one entry point
  and dispatch on the Client API task name.
- PyTorch swarm learning can now combine client-to-client tensor streaming with
  aggregation-client tensor disk offload. Set
  ``aggregation_format=ExchangeFormat.PYTORCH`` and
  ``enable_tensor_disk_offload=True`` on ``SwarmLearningRecipe``. The same
  offload flag is available on ``SwarmClientConfig`` for Job API users.
