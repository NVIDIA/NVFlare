.. _cc_architecture_overview:

Confidential Computing Architecture and Security
================================================

This page explains the trust model and protection boundaries of Federated
Confidential Computing with NVIDIA FLARE. For deployment selection and
preparation steps, start with :ref:`cc_getting_started`.

Responsibilities and trust boundaries
-------------------------------------

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

Workload protection across deployment modes
--------------------------------------------

Standalone Confidential Virtual Machines (CVMs) built with CVM Builder and
Confidential Containers (CoCo) extend the chain of trust to application content
in different ways. A compatible generic CVM can serve new application vaults
without rebuilding its verified root; each new vault still needs authorization.
CoCo binds an approved release to its encrypted image, guest policy, and launch
configuration. A runtime name or successful platform appraisal alone does not
prove that every application byte or output is authorized.

Attestation and workload authorization
--------------------------------------

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

Limits of the protection
------------------------

Confidential computing does not establish that approved code is safe, prevent
model inversion from authorized outputs, guarantee availability, or
automatically protect host-backed files. It also does not replace differential
privacy, cryptographic secure aggregation, application review, or organizational
agreements where those controls are required.

Side channels and replay of previously valid mutable storage require separate
consideration. CVM Builder's authenticated vault detects corruption but does not
provide general disk rollback protection. CoCo has deployment-specific
application-identity and guest-integrity limitations described in its
:ref:`security analysis <coco_security_attack_surface>`. Enabling a Trusted
Execution Environment (TEE) alone does not close these gaps or establish
regulatory compliance.

Deployment-specific architecture
---------------------------------

* :ref:`cc_architecture` describes the standalone CVM chain of trust, verified
  root, application vault, and attestation-bound key release.
* :ref:`coco_security_architecture` describes CoCo's trust boundaries,
  attestation and key-release protocol, operational roles, and residual risks.
* :doc:`azure/index` links to the Azure deployment guides and their
  platform-specific requirements.

Use :ref:`cc_getting_started` for the preparation workflow and
:ref:`cc_deployment` for the shared provisioning configuration.
