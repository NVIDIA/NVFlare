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

Kubernetes/OpenShift deployment, credential handling, and framework/recipe
additions also shipped this release; see `Also in This Release`_ below.

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

**Auto-FL research assistance** runs a coding-agent-directed campaign over
an existing job: NVFLARE owns the deterministic campaign import, execution,
policy boundaries, and provenance, while the agent proposes hypothesis-driven
candidates within the job's fixed training budget and mutation-schema
bounds. Its initial importer similarly supports statically recognizable
NVFLARE Recipe and ``*Job`` patterns.

All skills run behind a security-evaluation gate (prompt-injection and
untrusted-input eval suites) and include explicit safeguards for site-local
data and preprocessing. Install and invoke them through a coding agent as
described in :doc:`/user_guide/agent_skills/index` (see :ref:`autofl_skill`
for the Auto-FL workflow); start with the
:github_nvflare_link:`runnable Agent Skills examples
<examples/hello-world/agent-skills>` to try the conversion and
federated-statistics workflows.

Agent Skills are developer tooling, not a runtime FL API or a substitute for
code review and job validation: conversion skills preserve explicitly
supported Recipe and Client API paths, and ask or stop, rather than guess,
when framework ownership, source semantics, required data handling, or
runtime configuration is ambiguous.

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

   from nvflare.collab import CollabRecipe, collab
   from nvflare.recipe import SimEnv

   @collab.publish
   def train(model, update_type):
       updated_model = model + 1
       return updated_model, float(updated_model.mean())

   @collab.main
   def run():
       model = INITIAL_MODEL.copy()
       for _ in range(collab.get_app_prop("num_rounds", 3)):
           client_results = collab.clients.train(model, "full")
           # average client_results into the next round's model
       return model

   # CollabRecipe discovers the decorated functions above for both sides.
   CollabRecipe(job_name="hello_numpy_collab").execute(SimEnv(num_clients=2))

New examples:

- :github_nvflare_link:`Hello Collab <examples/hello-world/hello-collab>` —
  a minimal FedAvg workflow
- :github_nvflare_link:`pt_async_cifar10
  <examples/advanced/collab/pt_async_cifar10>` — FedBuff-style buffered
  asynchronous aggregation
- :github_nvflare_link:`advanced Collab examples <examples/advanced/collab>`
  — split learning, swarm learning, and in-time aggregation

Slurm Job Launcher
===================

FLARE 2.9.0 adds Slurm-native job execution for HPC environments — a new
execution target alongside the existing process, Docker, and Kubernetes
launchers. A long-lived NVFLARE parent submits each client or server job
process as a Slurm batch job; Slurm selects resources while FLARE manages the
federated job lifecycle. The launcher supports:

- Apptainer, Pyxis/Enroot, and bare-Python execution backends
- GPU-aware worker setup and multi-node applications
- a shared-file worker channel for clusters where compute nodes cannot open a
  direct connection to the parent

Follow the :ref:`slurm_job_launcher` deployment guide for prerequisites,
backend setup, site configuration, and validation steps.

Large-Model Training
=====================

FLARE 2.9.0 strengthens the streaming transport for long-running model
transfers.
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

- **Security and credential handling** — job-process bootstrap credentials
  move off command lines and into the process environment (a per-job
  Kubernetes Secret on K8s).

  - CLI and runtime diagnostics redact sensitive values more consistently.
  - Recipe APIs add safeguards for declaring and handling secrets; see
    :ref:`recipe_secrets`.

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

- The streaming transport retains its 16 MiB streaming-window and 4 MiB
  ACK-interval defaults for compatibility and bounded per-stream memory.
  High-bandwidth deployments may opt into larger values on all endpoints;
  ``dev_tools/f3/comm_config.yml`` provides a 64 MiB/16 MiB tuning example.
  TCP connections now enable ``TCP_NODELAY`` by default to reduce
  request/ACK latency.
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
