.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Overview
========

This document introduces the Confidential Computing (CC) attestation integration in NVFlare.

Please refer to the :ref:`NVFlare CC <confidential_computing>` for the introduction and detailed architecture of Confidential Computing.

Attestation enables participants to prove the integrity and trustworthiness of their computing environment. This mechanism ensures that only mutually trusted participants take part in a federated learning job, reinforcing both security and integrity across the NVFlare system.

How It Works
============

CC Token Generation
-------------------

Each participant uses a ``CCAuthorizer`` to generate a CC token that attests to its environment's security posture.

For example, the ``SNPAuthorizer`` utilizes AMD's ``snpguest`` utility to generate an attestation report and package it into a CC token.

CC Token Verification
----------------------

When a participant receives a CC token from another participant, it verifies the token's claims against its own security policy. This check ensures that the token owner is using the required hardware, software, and configurations to meet the security standards.

If verification fails—i.e., the CC token does not meet the policy—the site may choose not to participate in the job. It will not exchange models or collaborate further.

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

Configuration
-------------

You can configure the CC attestation components during the provision step. See the :ref:`NVFlare CC Deployment Guide <cc_deployment_guide>` for detailed instructions.

Runtime Behavior
================


The Confidential Computing (CC) attestation workflow establishes continuous, system-wide trust between all federated learning participants.

1. System Bootstrap
-------------------

When the system starts, each CC-enabled site (server or client) initializes its confidential computing components and generates a CC token that identifies its trusted environment.


2. Client Registration
----------------------

During client registration:

    - The client sends its token to the server.

    - The server verifies the client’s token and responds with its own.

    - The client then validates the server’s token.

This mutual verification ensures both sides trust each other before participating in any job.


3. Continuous Cross-Site Validation
-----------------------------------

After startup, all sites periodically perform cross-site token validation:

Each site generates new CC tokens at regular intervals.

Sites exchange tokens through a secure communication channel.

Every participant validates the tokens of all others.

If any CC-enabled site fails token validation, the system will shut down to maintain a trusted environment.
Sites that are not CC-enabled are skipped during attestation checks.


4. Job Scheduling
-----------------

Before jobs run, the server confirms that all CC-enabled participants have valid, verified tokens.
If validation fails, the system shuts down to prevent untrusted operation.
Jobs involving untrusted code (for example, BYOC) are blocked in CC mode.

5. Summary
----------

The attestation workflow provides:

    - Continuous, system-wide token verification

    - Mutual trust between server and clients

    - Automatic shutdown on attestation failure

This ensures that all confidential computing participants operate only within secure and attested environments.

