.. _confidential_computing_attestation:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

In NVFlare, data is encrypted during transmission between participants
—including the server, clients, and admin—to ensure strong protection
for data in transit. To secure data at rest, users can leverage existing
infrastructure such as storage encryption.

However, traditional security approaches leave a critical gap: **data in use**.
This is where confidential computing (CC) plays a key role. CC enables NVFlare
to protect data while it is being processed, completing full-lifecycle security:
at rest, in transit, and in use.

Confidential computing in NVFlare is also central to establishing trust between
participants in a federated learning job. Before participating in a job, each
participant must generate a set of cryptographically verifiable evidence about
its computing environment—including details about hardware (e.g., GPU),
software (e.g., GPU driver and VBIOS), and other platform components.

This evidence is validated, signed, and packaged into a Confidential Computing
token (CC token). The participant can then present its CC token to others to
prove the integrity and trustworthiness of its environment.

When a participant (the "relying party") receives a CC token, it verifies the
claims inside the token against its own security policy. This check ensures
that the token owner is using the required hardware, software, and
configurations to meet the relying party's security standards.

If the verification fails—i.e., the CC token does not meet the relying party's
policy—the relying party may choose to reject participation in the job. It will
not exchange models or collaborate further.

This mechanism ensures that only mutually trusted participants take part in a
federated learning job, reinforcing both security and integrity across the
NVFlare system.

We provided a CCManager component and several CCAuthorizer components for different hardware platforms.
Currently, we support the following CCAuthorizer components:
- SNPAuthorizer
- GPUAuthorizer
- ACIAuthorizer
- TDXAuthorizer

You can configure it using the provision step in the [cc deployment guide](cc_deployment_guide.rst).

****************
Runtime behavior
****************

When a participant—either the server or a client—starts up, the CCManager
responds to the EventType.SYSTEM_BOOTSTRAP event by generating its own
CC token using the configured CCAuthorizers.

When a client registers with the server, it includes its CC token as part
of the registration data. If the registration is successful, the server
collects and stores the client's CC token.

The server's CCManager maintains both its own CC token and the tokens of all
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
