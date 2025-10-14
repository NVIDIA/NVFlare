.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Please refer to the :ref:`NVFlare CC Architecture <cc_on_prem_cvm_architecture>`
for the introduction and detailed architecture of Confidential Computing.

This document introduces the CC attestation integration in NVFlare.

Each participant will use the corresponding ``CCAuthorizer`` to generate the CC token.

For example, the ``SNPAuthorizer`` utilizes AMD's ``snpguest`` utility to generate
an attestation report and package it into a CC token.

In NVFlare, the participant will first generate the CC token, then present its
CC token to others to prove the integrity and trustworthiness of its environment.

Upon receiving a CC token, the participant verifies its claims against its own
security policy. This check ensures that the token owner is using the required
hardware, software, and configurations to meet the security standards.

If verification fails—i.e., the CC token does not meet the policy—the site
may choose not to participate in the job. It will not exchange models or
collaborate further.

This mechanism ensures that only mutually trusted participants take part in a
federated learning job, reinforcing both security and integrity across the
NVFlare system.

We provide a ``CCManager`` component and several ``CCAuthorizer`` components for different hardware platforms.
Currently, we support the following ``CCAuthorizer`` components:

- ``SNPAuthorizer``
- ``GPUAuthorizer``
- ``ACIAuthorizer``
- ``TDXAuthorizer``

You can configure it using the provision step in the :ref:`NVFlare CC Deployment Guide <cc_deployment_guide>`.

****************
Runtime Behavior
****************

When a participant—either the server or a client—starts up, the ``CCManager``
responds to the ``EventType.SYSTEM_BOOTSTRAP`` event by generating its own
CC token using the configured ``CCAuthorizers``.

When a client registers with the server, it includes its CC token as part
of the registration data. If the registration is successful, the server
collects and stores the client's CC token.

The server's ``CCManager`` maintains both its own CC token and the tokens of all
registered clients.

Once a job is submitted and scheduled for deployment, the server verifies the
CC tokens of the clients listed in the job's deployment map, using its own
result policy.

If all client tokens in the deployment map pass verification, the server sends
the verified tokens to those clients for peer verification.

Each client then evaluates the received CC tokens against its own result policy
to decide whether it trusts the other participants. Based on this evaluation,
the client may choose to accept or reject participation in the job.

If a client declines to join the job, the server excludes it from deployment.

Finally, the server's job scheduler determines whether the job has sufficient
resources to proceed. It finalizes the job's status based on resource
availability and any defined retry policies.
