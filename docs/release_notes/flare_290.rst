:orphan:

**************************
What's New in FLARE v2.9.0
**************************

NVIDIA FLARE 2.9.0 expands the platform for research-oriented federated
workflows, AI-agent-assisted development, and large-scale training in HPC
environments. The release introduces the Collaboration API and Agent Skills,
adds Slurm-native job execution, and strengthens the client, recipe, streaming,
and operational foundations used for large-model workloads.

Release Highlights
==================

- **Collaboration API**: define collaborative workflows as ordinary Python
  functions or classes and package them with ``CollabRecipe``. The API removes
  much of the controller, executor, and payload boilerplate required by custom
  workflows while preserving explicit server/client behavior. New examples
  cover synchronous FedAvg, asynchronous PyTorch training, and split learning.
- **Agent Skills**: installable NVFLARE-owned skills give coding agents guided,
  reviewable workflows for converting PyTorch, PyTorch Lightning, and Hugging
  Face Trainer projects to federated jobs; producing federated statistics; and
  diagnosing generated jobs. The skills include source inspection, validation,
  packaging, and data-locality guardrails.
- **Slurm support for HPC**: run NVFLARE client and server job processes as
  Slurm allocations, with scheduler-managed submission, monitoring, and
  cancellation. Sites can use Apptainer, Pyxis/Enroot, or trusted bare-Python
  execution, and can use the shared-file worker channel where compute nodes
  cannot connect directly to the parent.
- **Kubernetes and OpenShift enhancements**: improve child-job failure
  propagation, pod-event visibility, OpenShift scheduling, container build
  guidance, and Docker/Podman compatibility for production deployments.
- **Large-model reliability and memory efficiency**: improve streamed model
  transfer handling, retry behavior, and bounded memory use. Tensor disk
  offload is available for PyTorch Swarm aggregation as well as FedAvg, and
  large blob transport has stronger failure handling and higher default limits.
  FedAvg now supports federated training of LLMs with up to 72 billion
  parameters when the deployment is sized and configured appropriately.
- **Framework and Recipe APIs**: federate Hugging Face ``Trainer`` and TRL
  ``SFTTrainer`` scripts through the Hugging Face Client API; keep a long-lived
  external trainer connected through Client API Attach mode; and use the new
  concrete PyTorch FedBPT recipe entry point.

Research and Developer Productivity
===================================

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

Large-Model Transport and Memory
--------------------------------

Streaming and tensor-transfer improvements make large-model workloads more
robust under slow links, retries, and delayed worker responses. PyTorch Swarm
can use tensor disk offload during aggregation, reducing peak memory pressure
by materializing incoming tensors through temporary disk storage. Operators
should continue to size timeouts, storage, and network connectivity for their
model size and deployment topology. With the corresponding infrastructure and
configuration, FedAvg supports federated LLM training at scales up to 72
billion parameters. See :ref:`notes_on_large_models` for deployment sizing and
large-model operational guidance.

Kubernetes and OpenShift Operations
-----------------------------------

Kubernetes and OpenShift deployments receive improved job-failure propagation,
pod-event access for diagnosis, and parent-pod scheduling behavior. Updated
container guidance covers Podman-based image builds alongside Docker, while
runtime compatibility updates improve Docker-in-Docker and development-tag
handling. Generated Helm Roles now include read-only Kubernetes Event access;
sites that maintain custom RBAC should add the equivalent Event read
permissions.

The OpenShift quickstart now requests ``500m`` CPU and ``1Gi`` memory per
parent pod by default, which fits the documented CRC configuration. Override
these values with ``PARENT_CPU`` and ``PARENT_MEMORY`` when preparing a larger
deployment. See :ref:`helm_chart` for the Kubernetes deployment workflow and
the :github_nvflare_link:`OpenShift example <examples/devops/openshift>` for
the OpenShift configuration and sizing guidance.

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
