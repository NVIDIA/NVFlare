.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Overview
========

This document introduces the shared Confidential Computing (CC) participant-attestation components in NVFlare.
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

For example, the ``SNPAuthorizer`` utilizes AMD's ``snpguest`` utility to generate an attestation report and package it into a CC token.

``CoCoAuthorizer`` instead uses the Kata guest-local attestation service to obtain Trustee's signed appraisal and creates
a guest-key proof for its NVFlare identity. Verification can run outside CoCo using the configured Attestation Service
public key; generation requires the protected guest services. See :ref:`coco_security_architecture` for that protocol.

CC Token Verification
----------------------

When a participant receives a CC token from another participant, it verifies the token's claims against its own security policy. This check ensures that the token owner is using the required hardware, software, and configurations to meet the security standards.

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

Each ``CCAuthorizer`` is responsible for generating attestation reports for a specific hardware platform. NVFlare provides multiple authorizers to support different confidential computing technologies.

Supported Platforms
===================

NVFlare currently supports the following ``CCAuthorizer`` components:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Authorizer
     - Platform
   * - ``SNPAuthorizer``
     - AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)
   * - ``GPUAuthorizer``
     - NVIDIA GPU Confidential Computing (H100, Blackwell)
   * - ``TDXAuthorizer``
     - Intel TDX (Trust Domain Extensions)
   * - ``ACIAuthorizer``
     - Azure Confidential Containers Instance
   * - ``CoCoAuthorizer``
     - Kata/Trustee-backed AMD SEV-SNP or Intel TDX, CPU-only or with NVIDIA GPU. TDX is experimental pending hardware acceptance; see the CoCo architecture.

Configuration
-------------

CVM boot and periodic vault reauthorization are configured through the CVM Builder profile and Trustee deployment. See
the :ref:`NVFlare CC Deployment Guide <cc_deployment_guide>` for that provisioning workflow. Application-level
``CCManager`` components in this CVM Builder path are configured separately in runtime resources.

The CoCo provisioning path generates ``CCManager`` and ``CoCoAuthorizer`` resources for protected clients and servers,
and for the ordinary participants that need verifier-only operation. It also provisions the required participant set
and identity-binding requirements. See :ref:`coco_security_architecture` and the
:github_nvflare_link:`CoCo CCManager configuration guide <examples/devops/coco/provision/CCMANAGER.md>`.

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

Mutual attestation requires both participants to be configured as protected. An ordinary server can verify CoCo clients
without attesting itself, and ordinary clients can verify a protected server without claiming their own TEE protection.


3. Continuous Cross-Site Validation
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
