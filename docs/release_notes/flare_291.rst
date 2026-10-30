:orphan:

**************************
What's New in FLARE v2.9.1
**************************

FLARE 2.9.1 updates the confidential computing deployment stack, which follows
the NVIDIA confidential computing reference architecture. The first version
was a technical preview in 2.7.0, and the stack remains a technical preview and
entirely optional: nothing changes for deployments that do not use it. The
other changes that can affect existing deployments are a Dashboard restriction
and a few bug fixes.

Highlights:

- **Confidential computing (technical preview)** — CVM Builder for Intel TDX
  and AMD SEV-SNP, Confidential Containers (CoCo) on Kubernetes, and one
  unified provisioning interface for bare-metal CVMs, CoCo, and Azure
- **Security hardening** — launcher-defined bring-your-own-code (BYOC) jobs
  blocked in confidential computing, checkpoint path confinement, and Dashboard
  client ``props`` rejected
- **Agent Skills** — published with a shared catalog bundle

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
  blocked in confidential computing, and checkpoint paths that escape the application root
  are rejected. CVM build inputs are validated more strictly, and the vault
  capacity is checked before a build.
- **Dashboard.** Client creation and PATCH requests that contain ``props`` are
  rejected with HTTP 400, and client-kit generation ignores previously stored
  ``props``, so client properties can no longer supply provisioning paths that
  include server-local files in downloadable kits.

Agent Skills
============

NVFlare skills are published with a shared catalog bundle, and the benchmark
provenance of the Auto-FL, PyTorch, Lightning, federated statistics, orient,
and diagnose-job skills is refreshed. Install the complete NVFlare skill set
together so the cross-references between skills resolve.

Bug Fixes
=========

- Admin connector inspection no longer times out when the receiver has not
  registered the internal ``DriverParams`` enum.
- Subprocesses start in a new session through a shared ``start_new_session``
  helper, replacing ``preexec_fn=os.setsid``.
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
- Confidential computing jobs can no longer carry launcher-defined BYOC
  content.
