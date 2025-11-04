.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Overview
========

This document introduces the Confidential Computing (CC) attestation integration in NVFlare.

Please refer to the :ref:`NVFlare CC Architecture <cc_architecture>` for the introduction and detailed architecture of Confidential Computing.

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

The attestation workflow consists of several phases during job lifecycle:

1. System Bootstrap
-------------------

When a participant (server or client) starts up, the ``CCManager`` responds to the ``EventType.SYSTEM_BOOTSTRAP`` event by generating its own CC token using the configured ``CCAuthorizers``.

2. Client Registration
----------------------

When a client registers with the server, it includes its CC token as part of the registration data. If the registration is successful, the server collects and stores the client's CC token.

The server's ``CCManager`` maintains both its own CC token and the tokens of all registered clients.

3. Job Deployment Verification
-------------------------------

Once a job is submitted and scheduled for deployment, the server verifies the CC tokens of the clients listed in the job's deployment map, using its own security policy.

If all client tokens in the deployment map pass verification, the server sends the verified tokens to those clients for peer verification.

4. Peer Verification
--------------------

Each client evaluates the received CC tokens (including the server's token and other clients' tokens) against its own security policy to decide whether it trusts the other participants.

Based on this evaluation, the client may choose to accept or reject participation in the job.

If a client declines to join the job, the server excludes it from deployment.

5. Job Scheduling
-----------------

Finally, the server's job scheduler determines whether the job has sufficient resources to proceed. It finalizes the job's status based on:

- Resource availability
- Number of participants that passed verification
- Any defined retry policies

This multi-stage verification process ensures that all participants in a federated learning job operate in trusted, attested environments.
