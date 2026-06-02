**************************
What's New in FLARE v2.8.0
**************************

NVIDIA FLARE 2.8.0 focuses on making production federated learning easier to
operate across organizations, studies, and runtime environments. The release
adds Docker and Kubernetes job launchers, a broader automation-friendly
CLI, distributed provisioning, multi-study support, stronger observability, and
additional production hardening. It also adds new examples and research bundles
for multimodal, language-model, Docker, Kubernetes, and privacy-oriented
federated learning workflows.

Release Highlights
==================

- **Modern NVFlare CLI**: expanded ``nvflare`` command groups for jobs,
  system operations, local config, startup kits, recipes, distributed
  provisioning, and deployment preparation, with JSON output and schema support
  so operators and automation systems can run FLARE workflows without relying
  on console-only behavior.
- **Distributed provisioning**: new ``nvflare cert`` and ``nvflare package``
  workflows let participants keep private keys local while Project Admins
  approve certificate requests and generate signed packages, improving security
  ownership in cross-organization deployments.
- **Deployment prepare and runtime packaging**: new ``nvflare deploy prepare``
  flow packages existing startup kits for Docker and Kubernetes runtimes,
  including Kubernetes environments on AWS, Azure, and GCP, so provisioning and
  runtime packaging can be handled as separate repeatable steps.
- **Docker and Kubernetes job launchers**: each site can configure a
  process, Docker, or Kubernetes job launcher. With the matching launcher
  configured, host-based jobs run as subprocesses, Docker-based jobs run as job
  containers, and Kubernetes-based jobs run as separate job pods, giving
  production sites Docker/Kubernetes isolation and resource handling plus
  study-scoped dataset mounts for stronger data isolation.
- **Multi-study support**: study definitions in ``project.yml``, study-scoped
  sessions, study-aware admin operations, and study CLI commands let one FLARE
  deployment host multiple collaborations without mixing participants,
  authorization, data access, or operational context.
- **Live log streaming**: site and job logs stream to the server while jobs are
  running, reducing time to diagnose remote training failures and making CLI
  automation more responsive.
- **Security and production hardening**: origin-bound auth tokens, safer archive
  handling, stricter private-key file permissions, safer loading paths, stronger
  job metadata validation, and additional dashboard/API hardening reduce common
  operational risk in federated deployments.
- **Feature election**: a new federated feature selection workflow lets clients
  perform local feature selection for tabular datasets and share feature scores,
  not raw data, so FLARE can aggregate a global feature mask for downstream
  training.
- **Tensor disk offload for FedAvg**: enabling ``enable_tensor_disk_offload=True``
  significantly reduces server peak memory during FedAvg aggregation. Instead of
  holding all client tensor updates in memory simultaneously, each update is written
  to a temporary safetensors file on disk and consumed lazily during aggregation.
  The benefit scales with model size and client count.
- **Large-model streaming reliability**: large tensor broadcasts are more robust
  when many clients retry after delayed EOF responses. Finished download refs are
  handled idempotently, and subprocess Client API jobs now reject unbounded result
  resends or missing download-completion waits that can turn one slow transfer into
  repeated large-model retries.
- **New examples and contributed research**: MedGemma, Qwen3-VL, Codon-FM,
  FedUMM, financial-services fraud detection, Docker job examples, distributed
  provisioning examples, Hello JAX, and Hello log streaming help teams start
  from working patterns instead of assembling production and research workflows
  from scratch.

NVFlare CLI and Automation
==========================

FLARE 2.8.0 significantly expands the public ``nvflare`` command-line surface.
The CLI now has a more consistent command layout, machine-readable output
support, and better error contracts for scripts and automation systems.

This matters for production operations because the same interfaces can now be
used consistently by humans, shell scripts, service automation, and other
tooling. Jobs, system status, startup-kit selection, recipes, provisioning, and
deployment preparation can be queried in structured form instead of requiring
manual console interaction.

The main additions are command groups for job operations, system operations,
local configuration, recipe discovery, distributed provisioning, package
assembly, and deployment preparation. Many commands now support structured
output and schema discovery, making them easier to use in scripts, notebooks,
and operational tooling.

For details, see :ref:`nvflare_cli`, :ref:`job_cli`, :ref:`system_command`,
:ref:`config_command`, and :ref:`recipe_command`.

For a hands-on CLI workflow, see the
:github_nvflare_link:`NVFlare CLI tutorial <examples/tutorials/nvflare_cli.ipynb>`.

Deployment and Provisioning
===========================

Distributed Provisioning
------------------------

FLARE 2.8.0 introduces a distributed provisioning workflow for cases where
participants generate local private keys and certificate-signing requests
instead of receiving all startup-kit materials from a centralized provisioner.

This is important for cross-organization collaborations because private keys no
longer need to be generated by the Project Admin or transferred between
organizations. Each participant can create and keep its own key material,
reducing key-handling risk, while the Project Admin still controls approvals,
signed packages, and root CA trust. Teams that prefer the existing centralized
provisioning model can continue to use it.

The workflow adds participant-side certificate requests, Project Admin approval,
signed package generation, root CA verification, and startup-kit assembly from
approved packages. It is intended for deployments where key ownership and
participant-controlled certificate requests are important.

See :ref:`distributed_provisioning`, :ref:`cert_command`, and
:ref:`package_command`.

For a runnable walkthrough, see the
:github_nvflare_link:`distributed provisioning example <examples/advanced/distributed_provision>`.

Deploy Prepare
--------------

The new ``nvflare deploy prepare`` command packages existing provisioned
startup kits for runtime targets such as Docker and Kubernetes. This separates
startup-kit generation from runtime-specific packaging, making deployments more
repeatable across local, Docker, Kubernetes, and cloud-managed Kubernetes
environments such as AWS, Azure, and GCP.

This separation is useful operationally because the same provisioned identities
can be reused across runtime-specific packaging flows. Teams can prepare a
startup kit once, then produce Docker or Kubernetes artifacts without changing
the provisioning model.

See the :ref:`deploy_prepare_command` user guide for Docker and Kubernetes
runtime preparation.

Docker and Kubernetes Job Execution
-----------------------------------

FLARE 2.8.0 adds Docker and Kubernetes job launchers so sites can align
FLARE jobs with the runtime isolation and resource controls they already use.
Each site must be configured with the matching job launcher for the intended
runtime. With that launcher configured, the pattern is:

- process job launcher for a host-based parent: jobs run as subprocesses;
- Docker job launcher for a Docker-based parent: jobs run as Docker
  containers;
- Kubernetes job launcher for a Kubernetes-based parent pod: jobs run as
  separate Kubernetes job pods.

This matters because Docker and Kubernetes deployments can now use their runtime
isolation instead of treating every job as a local subprocess. Study-dataset
mapping is also carried into containers and pods, so each job sees only the
datasets configured for its study scope, reducing cross-study data exposure.

Highlights:

- Kubernetes deployments can launch jobs in separate pods when configured with
  the Kubernetes job launcher.
- Docker deployments can launch jobs as separate containers when configured with
  the Docker job launcher.
- Study-dataset mappings provide study-scoped data isolation for Docker
  containers and Kubernetes job pods.
- CPU, memory, storage, and GPU requirements can be delegated to Docker or
  Kubernetes resource handling.
- Kubernetes job workspace transfer no longer depends on a shared job PVC.
- Runtime packaging, Helm chart updates, Docker job examples, multicloud
  Kubernetes support, and Brev scripted deployment guides make these modes
  easier to try and operate across AWS, Azure, and GCP environments.

For deployment details, see the :ref:`deploy_prepare_command` user guide and
the :ref:`helm_chart` Kubernetes deployment guide. Additional references include
:ref:`containerized_deployment`, :ref:`brev_deployment`, and
:ref:`brev_scripted_deployment`.

For a runnable Docker workflow using ``nvflare deploy prepare``, see the
:github_nvflare_link:`Docker job launcher example <examples/docker>`.

Multi-Study and Runtime Operations
==================================

Multi-Study Support
-------------------

FLARE 2.8.0 adds study-aware deployment and administration support. A single
deployment can define multiple studies, each with its own participating sites
and admin role mappings.

This is important for organizations that run more than one collaboration on the
same FLARE infrastructure. Study scope keeps participant membership,
authorization, admin sessions, and operational commands tied to the intended
collaboration, reducing the risk of cross-study confusion or accidental access.

The feature is intended to help one deployment support multiple projects or
consortia while keeping each study's participants, permissions, sessions,
commands, and data access scoped to that study. Study support is available in
administration workflows, CLI workflows, the FLARE API, production environments,
and local PoC development.

See :ref:`multi_study_guide` for design and configuration details, and
:ref:`study_command` for runtime management.

Live Log Streaming
------------------

FLARE can now stream job logs from clients to the server while the job is
running. Operators can inspect logs through server-side files or CLI commands
without waiting for the job to finish.

This shortens the feedback loop for production jobs, especially when training
runs remotely or for a long time. Operators can follow failures, progress, and
site-specific behavior while the run is active instead of waiting for final job
artifacts.

Operators can retrieve or follow job logs through the CLI and control log
streaming behavior at the site level. This is intended to make remote job
debugging and production monitoring less dependent on manual access to each
client machine.

See :ref:`live_log_streaming` and :ref:`site_config`.

For a runnable job example, see
:github_nvflare_link:`Hello log streaming <examples/hello-world/hello-log-streaming>`.

Recipes, APIs, and ML Features
==============================

2.8.0 continues the Recipe API and Client API direction from 2.7.x, with
additional workflow coverage and production fixes. These changes make recipe and
API-based workflows easier to automate, monitor, and operate in study-aware
environments.

Highlights include improved study-aware API behavior, better recipe run
management, updated Flower integration, stronger FedAvg and PyTorch workflow
handling, XGBoost and SVTPrivacy fixes, and Python support aligned to 3.10
through 3.14.

See :ref:`job_recipe`, :ref:`available_recipes`, :ref:`flare_api`, and
:ref:`api_evolution`.

For tutorial examples, see the
:github_nvflare_link:`Hello FLARE API notebook <examples/tutorials/flare_api.ipynb>`
and :github_nvflare_link:`Job Recipe notebook <examples/tutorials/job_recipe.ipynb>`.

Feature Election
----------------

FLARE 2.8.0 adds feature election, a federated feature selection workflow for
tabular datasets. Clients perform local feature selection and share selected
features and scores rather than raw data; FLARE aggregates the results into a
global feature mask that can be used for downstream federated training.

For a runnable workflow, see the
:github_nvflare_link:`feature election example <examples/advanced/feature_election>`.

Large Models and LLM Workflows
==============================

FLARE 2.8.0 builds on the large-model work from 2.7.2 with additional tensor
offload, run-scoped temp cleanup, improved timeout guidance for large transfers,
and new example coverage.

These improvements help large-model FL jobs operate under tighter memory and
runtime constraints, while the new examples give teams concrete starting points
for multimodal and language-model workloads.

Large-Model Streaming Reliability
---------------------------------

The streaming layer now treats late retries of normally finished download refs
as idempotent terminal responses instead of fatal missing-ref errors. This
addresses high-fanout large-model broadcasts where a client has completed a
download but retries because the final EOF response was delayed by network or
server-side contention.

The fix applies at the ``DownloadService`` layer, so it benefits large payload
transfers regardless of whether they come from FedAvg, Client API subprocess
jobs, tensor disk offload, or another feature built on the same streaming
path. Cleanup caused by transaction timeout or explicit deletion still returns
an invalid-ref error; only normally finished transactions are tombstoned for
late terminal retries.

Subprocess Client API jobs also validate risky retry settings earlier. In
particular, ``max_resends=None`` is now rejected for
``ClientAPILauncherExecutor`` jobs because unlimited resends can create an
unbounded sequence of large download transactions, and
``download_complete_timeout=None`` is rejected because the subprocess must stay
alive while the server finishes pulling tensors from it. Jobs with explicitly
configured large streaming request timeouts now receive warnings when related
pipe/download-completion timeouts are shorter than the configured streaming
timeout. Recipe-generated external-process jobs serialize the bounded
``max_resends=3`` default in executor args, and top-level
``recipe.add_client_config({"max_resends": N})`` overrides are applied before
the subprocess Client API config is written.

Server Memory: Tensor Disk Offload
-----------------------------------

FLARE 2.8.0 introduces tensor disk offload for PyTorch FedAvg jobs, which
significantly reduces peak server memory during aggregation. Instead of holding
all client tensor updates in memory simultaneously, each update is written to a
temporary safetensors file on disk and consumed lazily. The benefit scales with
model size and client count.

To enable, set ``enable_tensor_disk_offload=True`` on ``FedAvgRecipe`` or the
``FedAvg`` controller. This feature applies to PyTorch FedAvg workflows only.

.. warning::

   Temporary files use the server process temp directory (``TMPDIR`` or the OS
   default such as ``/tmp``). In containers or Kubernetes, ``/tmp`` is often
   RAM-backed (``tmpfs``), which eliminates the memory-saving benefit. The server
   admin must point ``TMPDIR`` to a disk-backed mount before starting the server.
   See :ref:`notes_on_large_models` for deployment guidance.

For configuration details, see :doc:`/programming_guide/tensor_downloader` and
:doc:`/programming_guide/memory_management`.

Corresponding examples include
:github_nvflare_link:`BioNeMo <examples/advanced/bionemo>`,
:github_nvflare_link:`Qwen3-VL <examples/advanced/qwen3-vl>`,
:github_nvflare_link:`MedGemma <examples/advanced/medgemma>`, and
:github_nvflare_link:`Codon-FM <examples/advanced/codon-fm>`.

Security and Hardening
======================

This release includes a broad set of security, validation, and operational
hardening changes.

The focus is reducing deployment risk in environments where jobs, startup kits,
archives, credentials, and admin/API traffic cross organizational boundaries.

Key areas include stronger runtime authentication binding, safer archive and path
validation, stricter private-key file permissions, safer deserialization and
subprocess handling, confidential-computing attestation hardening, dashboard/API
hardening, and clearer error behavior for admin and job operations.

Reliability and Bug Fixes
=========================

These changes improve day-to-day operability by making job state, startup,
resource visibility, and failure reporting more predictable across local,
Docker, Kubernetes, and server-connected workflows.

Notable improvements include more consistent job status publication, clearer
errors for missing or running jobs, more reliable startup and log-streaming
behavior, Docker and Kubernetes runtime fixes, better GPU visibility handling,
cleaner client failure reporting, corrected paired-duration monitoring metrics
when an end event is skipped, and refreshed integration-test and CI coverage.

New Examples and Research
=========================

2.8.0 adds or updates a wide range of examples and contributed research
implementations.

These assets matter because they turn new platform capabilities into runnable
starting points for teams evaluating FLARE in concrete domains, including
containerized operations, multimodal models, financial services, and
privacy-oriented research.

Research updates in 2.8.0 include:

- :github_nvflare_link:`FedUMM <research/fedumm>`: a new federated learning
  implementation for unified multimodal models, using parameter-efficient LoRA
  adapter federation for multimodal foundation-model workflows.
- :github_nvflare_link:`financial-services fraud detection <research/fsi-fraud-detection>`:
  a new privacy-preserving federated fraud detection implementation with
  synthetic payment transaction generation, heterogeneous site configurations,
  federated analytics, federated training, interpretability, and differential
  privacy experimentation.
- Existing :github_nvflare_link:`FedBPT <research/fed-bpt>` research was
  updated with a Job API entry point for running and exporting FLARE jobs.

Examples and research assets include:

- :github_nvflare_link:`Hello JAX <examples/hello-world/hello-jax>`.
- :github_nvflare_link:`Hello log streaming <examples/hello-world/hello-log-streaming>`.
- :github_nvflare_link:`Docker job execution <examples/docker>`.
- :github_nvflare_link:`distributed provisioning <examples/advanced/distributed_provision>`.
- :github_nvflare_link:`feature election <examples/advanced/feature_election>`.
- :github_nvflare_link:`MedGemma <examples/advanced/medgemma>`.
- :github_nvflare_link:`Qwen3-VL <examples/advanced/qwen3-vl>`.
- :github_nvflare_link:`Codon-FM <examples/advanced/codon-fm>`.
- :github_nvflare_link:`FedUMM <research/fedumm>`.
- :github_nvflare_link:`financial-services fraud detection <research/fsi-fraud-detection>`.

Compatibility and Migration Notes
=================================

- Python 3.9 is no longer listed as a supported development target. FLARE 2.8.0
  targets Python 3.10, 3.11, 3.12, 3.13, and 3.14.
- The deprecated FLAdminAPI surface has been removed. Use the FLARE API,
  Recipe environments, and ``nvflare`` CLI workflows for new automation.
- HA/Overseer code has been removed from the 2.8 branch.

See the :ref:`migration_guide` for additional API and configuration migration
notes.

Getting Started
===============

To explore the new 2.8.0 workflows:

- start with :ref:`quickstart` for a basic FLARE run.
- use :ref:`nvflare_cli` for the current CLI command surface.
- use :ref:`distributed_provisioning` for participant-managed certificates and
  signed startup-kit packaging.
- use :ref:`deploy_prepare_command` for Docker and Kubernetes runtime
  packaging.
- use :ref:`multi_study_guide` for multi-tenant deployment configuration.
- browse :ref:`available_recipes` and the new examples under
  ``examples/hello-world`` and ``examples/advanced``.
