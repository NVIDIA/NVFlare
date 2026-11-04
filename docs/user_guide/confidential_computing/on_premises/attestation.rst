.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Overview
========

This document introduces the shared Confidential Computing (CC) participant-attestation components in NVFlare.
For the shared CVM/CoCo peer-proof boundaries, read :ref:`cc_peer_attestation_limits`.
For the complete CoCo-specific trust model, protocol, failure behavior, and limitations, read
:ref:`coco_security_architecture`. CVM Builder's boot and vault reauthorization are separate mechanisms described in
:ref:`cc_architecture`.

Please refer to the :ref:`NVFlare CC <confidential_computing>` for the introduction and detailed architecture of Confidential Computing.

Attestation lets participants verify evidence about their peers against configured requirements. Its guarantees depend
on the authorizer, the required participant set, and the local verification policy; it does not establish that an
attested application behaves benignly or that its outputs preserve privacy.

How It Works
============

CC Token Generation
-------------------

A protected participant uses a ``CCAuthorizer`` to generate a CC token for peer verification. A verifier-only participant
can verify protected peers without generating evidence for its own environment.

Unified CVM and CoCo provisioning selects ``TrusteeAuthorizer``, which uses the
``CoCoAuthorizer`` proof implementation. Its CVM provider reads credentials from
the protected guest runtime; its CoCo provider uses the Kata guest-local AA
token API. The authorizer checks Trustee's signed appraisal and creates a proof
signed with the corresponding guest-held key. Verification uses the configured
Attestation Service public key and can run at an ordinary, verifier-only site.

The standalone ``SNPAuthorizer``, ``GPUAuthorizer``, and ``TDXAuthorizer``
classes are :ref:`legacy implementations <cc_legacy_authorizers>` and are not
selected by the unified CVM/CoCo path. The SNP/GPU authorizers' prover-generated
nonce histories are replay checks, not the Trustee signed-site proof or a
verifier-issued CC challenge. Do not assume every authorizer has the same
identity-binding guarantees.

CC Token Verification
----------------------

Verification evaluates the claims and checks implemented by the configured
authorizer and policy. For Trustee proofs, ``CCManager`` uses
``verify_for_site()`` to compare the signed site name with the independently
authenticated FL identity, and the authorizer checks the project audience,
signatures, freshness, and configured workload constraints. Proof validity alone
does not establish an exact running application or its behavior.

Trustee peer proofs are not bound to a particular TLS session and contain no
verifier-issued CC challenge. They do not replace normal FL PKI authentication
or protection of endpoint credentials. The trusted guest services' key-release
protocol is separate from this federation peer-proof protocol.

Failure handling is enforced by ``CCManager``, not an optional per-job choice. An invalid client registration is rejected.
A required-peer failure during cross-site validation invokes server-wide shutdown on the server or exits the detecting
client; the current behavior is not selective quarantine of an offending peer.

Components
==========

CCManager
---------

The ``CCManager`` component orchestrates the attestation process across the NVFlare system. It is responsible for:

- Generating CC tokens for the local participant
- Collecting and storing CC tokens from other participants
- Verifying tokens against security policies
- Coordinating peer verification among participants

CCAuthorizer
------------

Each ``CCAuthorizer`` implements token generation and verification for its
provider. Verifier-only instances need no local TEE, and a token can include a
still-valid cached appraisal. NVFlare provides multiple authorizers with
different evidence and identity-binding contracts.

Available Authorizers
=====================

NVFlare provides the following ``CCAuthorizer`` components. Availability does
not imply identical proof guarantees or hardware qualification for every target:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Authorizer
     - Platform or role
     - Status
   * - ``TrusteeAuthorizer``
     - Shared Trustee peer-proof format with deployment-specific guest providers.
     - Selected by unified bare-metal CVM and CoCo provisioning.
   * - ``SNPAuthorizer``
     - AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)
     - :ref:`Legacy <cc_legacy_authorizers>`; candidate for deprecation or removal.
   * - ``GPUAuthorizer``
     - NVIDIA GPU Confidential Computing (H100, Blackwell)
     - :ref:`Legacy <cc_legacy_authorizers>`; candidate for deprecation or removal.
   * - ``TDXAuthorizer``
     - Intel TDX (Trust Domain Extensions)
     - :ref:`Legacy <cc_legacy_authorizers>`; candidate for deprecation or removal.
   * - ``ACIAuthorizer``
     - Azure Confidential Containers Instance
     - Azure integration; separate token semantics.
   * - ``CoCoAuthorizer``
     - Kata/Trustee-backed SNP or TDX, CPU-only or with NVIDIA GPU.
       See :ref:`coco_security_architecture` for target profiles and
       :ref:`coco_security_acceptance` for recorded acceptance evidence and limits.
     - Shared proof implementation used by ``TrusteeAuthorizer``.

.. _cc_legacy_authorizers:

Legacy standalone authorizers
------------------------------

``GPUAuthorizer``, ``TDXAuthorizer``, and ``SNPAuthorizer`` are legacy standalone
implementations retained in 2.9.1. They are candidates for deprecation or
removal; a release for that change has not been selected. They are not
recommended for new deployments. Use the unified CVM/CoCo provisioning path
with ``TrusteeAuthorizer`` for the supported target profiles described in
:ref:`cc_deployment_guide`.

Existing deployments that explicitly configure these classes should plan a
migration to the appropriate unified profile and validate its evidence,
workload policy, and qualification requirements. The legacy classes do not
provide the same proof contract as ``TrusteeAuthorizer``; changing a class
name alone is not a migration.

Configuration
-------------

CVM boot and periodic vault reauthorization are configured through the CVM
Builder profile and Trustee deployment. See :ref:`cc_deployment_guide` for that
workflow.

Unified provisioning generates ``CCManager`` and ``TrusteeAuthorizer`` resources
for protected CVM/CoCo participants and for ordinary participants that need
verifier-only operation. It also configures the required participant set and
each site's required proof namespaces. See :ref:`coco_security_peer_attestation`
for the shared proof format and :doc:`../deployment` for configuration.

Runtime Behavior
================


The workflow below describes ``CCManager`` participant checks. These are point-in-time and periodic checks, not
continuous hardware monitoring or per-operation authorization. An authorizer can reuse a still-valid signed appraisal;
a new peer proof does not necessarily mean a new hardware quote was collected.

1. System Bootstrap
-------------------

At bootstrap, each configured site initializes its authorizers and verifiers. Tokens are generated when needed for
registration and cross-site validation; verifier-only sites do not claim a protected local environment.


2. Client Registration
----------------------

During client registration:

    - A protected client sends its token to the server.

    - The server verifies tokens required for the authenticated client identity and responds with its own token if the server is protected.

    - The client validates the server's token when the server is in its locally configured required participant set.

The server rejects an invalid protected-client proof during registration. A
client checks a required protected server's proof after registration and exits
on failure before starting protected work.

Mutual attestation requires both participants to be configured as protected. An ordinary server can verify CoCo clients
without attesting itself, and ordinary clients can verify a protected server without claiming their own TEE protection.


3. Periodic Cross-Site Validation
-----------------------------------

After startup, sites with ``CCManager`` periodically perform cross-site token validation:

Each protected site generates peer tokens when requested, subject to its authorizer's freshness and caching rules.

Sites exchange tokens through a secure communication channel.

Each verifier validates the locally required participants. Discovery supplies peer routes, not permission to remove
participants from the required set.

Missing or invalid required-peer evidence causes the detecting server to initiate system shutdown or the detecting
client to exit. An unreachable required participant can therefore interrupt the federation. Participants explicitly
outside the required set are not required to produce evidence; that is a configured trust choice, not attestation success.


4. Job Scheduling
-----------------

The server performs an initial cross-site validation when scheduling first checks client resources, unless a periodic
validation has already run. It does not request a new cross-site validation before every job; subsequent checks follow
the configured periodic schedule. A validation failure invokes the shutdown behavior described above.
Jobs involving untrusted code (for example, BYOC) are blocked in CC mode.

5. Summary
----------

The attestation workflow provides:

    - Registration, initial scheduling-related, and periodic participant verification

    - Mutual attestation when both server and clients are configured as protected

    - Registration rejection or shutdown/exit on required-peer validation failure, depending on the phase

These checks enforce configured participant requirements. They do not replace workload authorization, secure storage,
network authentication, application review, or the separate CoCo/KBS and CVM Builder key-release mechanisms.
