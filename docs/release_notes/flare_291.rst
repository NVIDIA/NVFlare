**************************
What's New in FLARE v2.9.1
**************************

FLARE 2.9.1 updates the confidential computing deployment stack, which follows
the NVIDIA confidential computing reference architecture. The first version
was a technical preview in 2.7.0, and the stack remains a technical preview and
entirely optional. Changes affecting other deployments include stricter
control-request authorization, checkpoint path and Dashboard restrictions,
removal of the legacy push-based tensor-streaming API, updated security
dependencies, configuration-format deprecations, and reliability fixes.

Highlights:

- **Confidential computing (technical preview)** — CVM Builder for Intel TDX
  and AMD SEV-SNP, Confidential Containers (CoCo) on Kubernetes, and one
  unified provisioning interface for bare-metal CVMs, CoCo, and Azure
- **Security hardening** — launcher-defined bring-your-own-code (BYOC) jobs
  blocked in confidential computing, approved Azure Container Instances (ACI)
  policies and signed external startup kits enforced, control requests bound
  to authenticated callers and jobs, checkpoint path confinement, and
  Dashboard client ``props`` rejected
- **Tensor transfer API** — legacy push-based tensor streaming removed; use
  the pull-based Tensor Downloader for large-model transfer
- **Configuration formats** — HOCON/PyHOCON and OmegaConf-based configuration
  are deprecated in 2.9.1 and will be removed in 2.10.0. Use JSON for runtime
  and job configuration.

Confidential Computing
======================

.. admonition:: Technical Preview

   Confidential computing is optional and a technical preview. Supported
   configuration targets do not imply hardware acceptance for every platform,
   software revision, or workload. Consult the validation status of the
   selected deployment before approving it for use.

NVIDIA Confidential Computing Reference Architecture
----------------------------------------------------

The new confidential computing stack follows the NVIDIA confidential computing
reference architectures, for both Confidential Containers (CoCo) and
self-hosted confidential VMs (CVM Builder). See
:ref:`cc_architecture_overview` for how FLARE maps its components, trust
boundaries, and attestation flow onto them, and for links to the NVIDIA
reference architecture documents.

Unified Provisioning Interface
------------------------------

All confidential participants are provisioned through one interface. Each
protected participant selects a deployment mode in its ``cc_config``:
``bare_metal_cvm``, ``coco``, or ``azure_cc``, and the project supplies one
``cc_project_config`` that owns the shared Trustee or Azure attestation
settings, registries, approval keys, and trusted build tools. A common
``CCBuilder`` and ``CCPackager`` verify signed startup kits, move plaintext
kits to private state, and publish a delivery manifest for each participant
(``cc_manifests/<participant>.json``). Modes can be mixed in one federation.

CVM Builder (Bare-Metal CVMs)
-----------------------------

CVM Builder creates reusable, approved generic confidential VMs and combines
them with an application-specific encrypted vault during provisioning.

- **Two stages.** Stage 1 builds and approves a generic CVM once per CPU/GPU
  profile. Stage 2 seals an application's Docker image, code, configuration,
  and secrets into a separate authenticated vault, so application updates
  reuse the approved generic CVM.
- **Intel TDX and AMD SEV-SNP**, CPU-only or with NVIDIA confidential GPUs.
  GPU profiles require a composite CPU and GPU authorization before the vault
  key is released.
- **Measured boot and attestation-bound key release** through an upstream
  Trustee service, with approved measurements in its reference-value storage,
  periodic re-authorization, workload quarantine, and forced shutdown on
  security failures.
- Delivered as OCI artifacts that can be published and pulled.

Confidential Containers (CoCo)
------------------------------

CoCo runs NVFlare participants as Kubernetes workloads inside Kata
confidential VMs, with encrypted workload images and guest policy bound to
measured InitData. It supports Intel TDX and AMD SEV-SNP, CPU-only or with
NVIDIA confidential GPUs, and mixed TDX and SNP participants. Peers verify
each other with identity-bound proofs checked against the Trustee attestation
service (``CoCoAuthorizer``). The example workflow starts from an existing
CoCo-ready cluster and Trustee service.

Azure Confidential Computing
----------------------------

Azure confidential VM and confidential container instance guides are
refreshed. The Azure integration accepts AMD SEV-SNP without a GPU.

For the supported single-ACI-server deployment, provisioning now binds signed
attestation evidence to the exact administrator-approved Confidential Computing
Enforcement (CCE) policy. The fixed server image verifies a separately delivered
startup-kit archive using a deployment-signing key before checking the kit
signatures and starting the server. Existing ACI deployments require
reprovisioning; there is no signature-only fallback. See
:ref:`aci_migration_291` for the required inputs and deployment order.

Documentation
-------------

New getting-started, architecture, CoCo security architecture, and unified
deployment reference pages, including the hardware evidence limits of the
validation runs.

Security and Hardening
======================

- **Confidential computing.** Launcher-defined bring-your-own-code (BYOC) jobs,
  which carry their own executable content in Docker, Kubernetes, or Slurm
  launcher specifications instead of using the approved workload image, are
  blocked in confidential computing. CVM build inputs are validated more
  strictly, and the vault capacity is checked before a build.
- **PyTorch checkpoint paths (all deployments).** ``PTFileModelPersistor``
  requires a checkpoint directory relative to and within the application
  directory, and checkpoint filenames relative to the checkpoint directory
  that do not traverse above it. Resolved checkpoint paths must stay within
  the application directory, including through symlinks. Absolute paths and
  paths that resolve outside the application directory are rejected. See the
  :ref:`migration_guide` for required configuration changes.
- **Control-request authorization (all deployments).** Client administration
  and job-control requests are checked against the actual sending endpoint
  and allowed upstream route. Server-parent commands are scoped to the job
  and caller role: only the matching server job can obtain full participant
  data or update its run status. A same-job client recovery probe receives
  existence information without participant credentials. Server-job controls
  require the parent server, and job bootstrap credentials are bound to the
  matching job or workspace-transfer identity.
- **Multiprocess component authorization.** Runtime component construction in
  sub-workers consults the parent site's authorization callback and rejects
  builds if authorization fails or is unavailable.
- **Client log streaming.** Log ownership is bound to the authenticated sender;
  invalid path components and symlinks are rejected. Default managed staging
  moves from the system temporary directory to
  ``{workspace.get_log_root()}/.client_log_streams/{job_id}/{client_name}/{log_file_name}``,
  keeps active logs available across job cleanup, and uses durable deletion
  intent to coordinate stream publication with explicit job deletion. Custom
  log collectors must follow the new path, and custom storage providers need
  authoritative presence checks for deletion recovery; see the
  :ref:`migration_guide`.
- **Dependency updates.** Package requirements now include
  ``certifi>=2026.10.19`` and ``uv>=0.13.6,<0.14.0`` following the
  eMudhra/emSign CA incident. Refresh locked dependencies and installed
  environments when upgrading.
- **Dashboard.** Client creation and PATCH requests that contain ``props`` are
  rejected with HTTP 400, and client-kit generation ignores previously stored
  ``props``, so client properties can no longer supply provisioning paths that
  include server-local files in downloadable kits.

Removed Tensor Streaming API
============================

The legacy push-based ``nvflare.app_opt.tensor_stream`` package, including
``TensorServerStreamer`` and ``TensorClientStreamer``,
``Recipe.enable_tensor_streaming()``, and the ``examples/advanced/tensor-stream``
example are removed. Jobs referencing these components fail to load; recipe
scripts calling the removed method fail as well. Migrate to the pull-based
:ref:`tensor_downloader` as described in :ref:`tensor_transfer_migration_291`.
Tensor Downloader, its timeout settings, and tensor disk offload remain supported.

Deprecations
============

HOCON configuration through PyHOCON (``.conf``) and YAML configuration through
OmegaConf (``.yml`` and ``.yaml``), including their ``.default`` variants, are
deprecated in 2.9.1. They remain supported in 2.9.1 and will be removed in
2.10.0. Use JSON for new runtime and job configurations, and migrate existing
files before upgrading to 2.10.0. See
:ref:`configuration_format_migration_291` for the migration steps.

This deprecation concerns the runtime and job configuration loaders. It does
not cover provisioning YAML files such as ``project.yml`` and
``cc_project.yml``, which use separate provisioning loaders.

Bug Fixes
=========

- Admin connector inspection no longer times out when the receiver has not
  registered the internal ``DriverParams`` enum.
- Subprocesses start in a new session through a shared ``start_new_session``
  helper, replacing ``preexec_fn=os.setsid``.
- In-time FedAvg aggregation stops on an accumulation exception before
  updating or saving a partially accumulated global model, preserving the
  previous checkpoint.
- Asynchronous gRPC startup accepts the ``agrpc``/``agrpcs`` scheme pair for
  direct and relayed clients.
- Recipe client log streaming (``enable_log_streaming``) works again in server
  jobs. The server-job auxiliary handler now passes the authenticated sender to
  the log receiver, so incoming client log chunks are no longer rejected for a
  missing client identity.
- Multiprocess sub-workers receive the required initialization services and
  report initialization failures as fatal instead of continuing with missing
  state.
- Confidential computing:

  - GPU CVMs treat quarantine as terminal after Ready-state revocation (the
    guest powers off); quarantine ordering is fixed, and the vault is revoked
    on a best-effort basis when quarantine itself fails. CPU-only profiles
    keep bounded recovery.
  - Cold-boot clock synchronization is given a longer window so AMD SEV-SNP
    bootstrap passes the clock-quality gate.
  - The portable AMD SEV-SNP guest policy is the default.
  - Cross-vault attestation policy selection and the CVM Builder client
    executor allow list are fixed.
  - Several of these change the measured image or profile version, so rebuild
    and re-approve the generic CVM, and rerun hardware acceptance, after
    upgrading.
- Tests that need Git metadata are skipped when it is unavailable.

Documentation and Examples
==========================

- Hello World examples and their requirements are updated for NVFlare 2.9.1.
- The documentation build works with Sphinx 8.
- The CoCo executor limitation and the in-process executor workaround are
  documented.

Upgrade and Compatibility
=========================

See the :ref:`migration_guide` for details.

- Confidential computing configuration changed. The earlier technical-preview
  provisioning syntax (introduced in 2.7.0), packager entry points, and
  HashiCorp Vault key path are replaced; there is no in-place upgrade of an
  existing deployment, and the change affects only deployments that use
  confidential computing.
- Dashboard API integrations that send client ``props`` must omit it.
- ``PTFileModelPersistor`` configurations with absolute or escaping checkpoint
  output paths must use paths within the application directory instead, in all
  deployments.
- Confidential computing jobs can no longer carry launcher-defined BYOC
  content.
- Existing ACI deployments must supply the prepared CCE policy and a separate
  deployment-signing key, then reprovision and deliver the signed external kit.
- Jobs using the removed push-based tensor-streaming components or recipe
  method must migrate to Tensor Downloader.
- Custom integrations using internal control channels must follow the caller,
  route, and job restrictions, including the server-parent heartbeat check;
  supported admin and job APIs retain their flows.
- Multiprocess sub-workers enforce the site's component authorization for
  runtime builds. A veto, unavailable authorization, or initialization failure
  stops execution.
- In-time FedAvg now terminates on an accumulation exception and preserves
  the previous checkpoint instead of saving a partially accumulated model.
- Custom live-log collectors must follow the new managed staging location;
  custom log producers must preserve the authenticated sender identity.
- Dependency locks and offline package mirrors must satisfy the updated
  ``certifi`` and ``uv`` requirements.
