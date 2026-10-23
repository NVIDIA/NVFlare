.. _confidential_computing:

###############################
FLARE Confidential Federated AI
###############################

.. admonition:: Technical Preview

   FLARE Confidential Federated AI is a **technical preview**. Supported
   configuration targets do not imply hardware acceptance for every platform,
   software revision, or workload. Consult the selected deployment's validation
   status before approving it for use.

Federated learning keeps training data at participating sites, but models and
updates still cross organizational boundaries. Model owners, data owners, and
infrastructure operators can have different interests and administrative rights.

FLARE combines hardware-backed confidential execution with participant
attestation and deployment-specific workload authorization. An approved
deployment can protect code, model weights, and data in use from the host
operator. Its guarantees also depend on trusted guest software, application
behavior, storage, and the policy governing released outputs.

Three protection goals
======================

**Build explicit trust between participants.** A TLS certificate identifies a
participant; it does not establish the state of its execution environment.
Attestation lets participants verify signed evidence against configured
requirements. Workload authorization determines which approved application may
receive a particular secret. These checks serve different purposes.

**Protect aggregation at the server.** A protected server processes individual
model updates inside an approved Trusted Execution Environment (TEE). The
aggregation application can still see those updates, while the host operator
stays outside the protected execution boundary. This is distinct from a
cryptographic secure-aggregation protocol. Authorized aggregate outputs can
still reveal information and require their own privacy policy.

**Protect training and model IP at client sites.** A model owner can supply
approved code and weights for execution on a data owner's confidential compute.
The data owner controls access to local data. Neither party should assume that
a TEE makes the application benign: the approved application can read its inputs
and may disclose them through permitted outputs.

Responsibilities and trust boundaries
=====================================

.. list-table:: Roles in a confidential federation
   :header-rows: 1
   :widths: 28 72

   * - Role
     - Responsibility
   * - Model or workload owner
     - Reviews application code and dependencies, approves the workload release,
       and defines permitted model and result access.
   * - Data owner
     - Authorizes which workload may access local data and reviews the workload's
       behavior and output policy.
   * - Trust-service administrator
     - Manages trusted reference values, appraisal policy, and secret-release
       authorization. Participants must trust this authority.
   * - Infrastructure operator
     - Runs the host or cluster. The confidential deployment treats this
       infrastructure as untrusted for protected plaintext, but cannot prevent
       the operator from denying service.

One organization may perform multiple roles. The deployment must still preserve
the intended separation of authority. Giving a host operator control of key
release or approved reference values changes the trust model.

Deployment choices in NVFlare 2.9
=================================

All three modes use the unified :ref:`cc_deployment` interface. Select a mode
for each protected participant through ``cc_config`` and define shared services
in ``cc_project_config``. A federation may mix deployment modes and include
ordinary, verifier-only participants. An ordinary participant can verify peers
without claiming that its own environment is confidential.

.. list-table:: Deployment forms and protection mechanisms
   :header-rows: 1
   :widths: 24 40 36

   * - Deployment mode
     - Protected workload and storage
     - Attestation and documentation
   * - ``bare_metal_cvm``
     - Standalone Confidential VM (CVM) with a measured, verified generic root
       and a separately sealed, authenticated application vault.
     - Trustee authorizes vault-key release. Start with :ref:`cvm_builder` and
       the :ref:`CVM security architecture <cc_architecture>`.
   * - ``coco``
     - Kubernetes workloads inside Kata confidential VMs, using encrypted
       workload images and InitData-bound guest policy.
     - Trustee authorizes release resources. See :ref:`coco_security_architecture`
       for the security model, role-specific procedures, and remaining gaps.
   * - ``azure_cc``
     - Azure confidential VMs or confidential container instances. External
       Azure tooling owns image and resource creation.
     - Microsoft Azure Attestation integrates with NVFlare participant checks.
       See :doc:`azure/index` for deployment-specific requirements.

CVM Builder and CoCo provide different ways to extend the chain of trust to
application content. A compatible generic CVM can serve new application vaults
without rebuilding its verified root; each new vault still needs authorization.
CoCo binds an approved release to its encrypted image, guest policy, and launch
configuration. A runtime name or successful platform appraisal alone does not
prove that every application byte or output is authorized.

The 2.9 configuration supports AMD SEV-SNP and Intel TDX targets for standalone
CVM and CoCo deployments, with CPU-only or NVIDIA confidential-GPU profiles.
**CoCo TDX remains experimental pending end-to-end hardware acceptance.**
For standalone CVMs, consult the :github_nvflare_link:`revision-specific
validation record <nvflare/lighter/cc/image_builder/VALIDATION.md>` and
:github_nvflare_link:`conformance requirements
<nvflare/lighter/cc/image_builder/CONFORMANCE.md>`.
The Azure integration currently accepts AMD SEV-SNP with ``gpu_tee: none``.

Attestation and workload authorization
======================================

Boot-time secret release and federation participation are separate decisions:

* **Deployment authorization** checks approved platform evidence and the
  deployment's resource or workload binding before releasing secrets. CPU-only
  profiles do not require GPU evidence. GPU-required CVM and CoCo profiles
  require successful CPU and GPU appraisal before the relevant key release.
* **Participant verification** uses ``CCManager`` and its authorizers to check
  peers during registration and periodic cross-site validation. Unified CVM and
  CoCo provisioning generates ``TrusteeAuthorizer`` components that use the same
  identity-bound peer-proof format through different guest integrations.

Peer checks are point-in-time and periodic, not continuous hardware monitoring
or a fresh appraisal before every operation. A valid peer proof is not a dataset
access-control policy or, by itself, proof of an exact running application image.
An invalid registration is rejected. A required-peer failure during cross-site
validation shuts down the server or exits the detecting client; it does not
selectively quarantine only the offending participant. See
:ref:`confidential_computing_attestation` for the runtime behavior.

Within one trusted administration domain, standalone CVM profiles and CoCo can
share a Trustee instance with separate appraisal policies and reference values.
This does not provide administrative isolation for mutually untrusted tenants.
See the :github_nvflare_link:`Trustee configuration guide
<nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md>`.

Workload and data-handling requirements
=======================================

* Package reviewed application code and dependencies in the approved workload
  image. CC mode blocks BYOC jobs, including launcher-defined executable content.
  Use exact, reviewed ``class_allow_list`` entries for additional components.
* Validate the job's executor against the selected deployment. External-process
  execution has failed in sealed CoCo deployments; follow the documented
  :github_nvflare_link:`in-process executor workaround
  <examples/devops/coco/FL-DEPLOYMENT.md#executor-limitation-and-workaround>`.
  Changing code or policy requires a new approved release.
* Keep confidential checkpoints, credentials, and logs within protected storage
  or an explicitly approved secure output path. CVM Builder's ``user_config``,
  ``user_data``, and ``applog`` sidecars are clear storage, not secret stores.
* Define who may receive datasets, individual updates, final models, and logs.
  An encrypted image does not implement dataset authorization or output review.
* Treat GPU driver and VBIOS versions as approved policy inputs. See the
  :github_nvflare_link:`CoCo GPU reference-value procedure
  <examples/devops/coco/service/GPU-REFERENCE-VALUES.md>` before deploying or
  changing a GPU baseline.

Limits of the protection
========================

Confidential computing does not establish that approved code is safe, prevent
model inversion from authorized outputs, guarantee availability, or
automatically protect host-backed files. It also does not replace differential
privacy, cryptographic secure aggregation, application review, or organizational
agreements where those controls are required.

Side channels and replay of previously valid mutable storage require separate
consideration. CVM Builder's authenticated vault detects corruption but does not
provide general disk rollback protection. CoCo has deployment-specific
application-identity and guest-integrity limitations described in its
:ref:`security analysis <coco_security_attack_surface>`. Enabling a TEE alone
does not close these gaps or establish regulatory compliance.

Getting started
===============

1. Identify which servers and clients need protection, who owns each trust
   decision, and which outputs may leave the protected environment.
2. Choose a deployment mode and review its architecture and hardware-validation
   status.
3. Follow :ref:`cc_deployment` for provisioning, then the selected mode's build,
   approval, launch, and operational procedures.

.. toctree::
   :maxdepth: 2

   deployment
   coco_security_architecture
   on_premises/index
   azure/index
