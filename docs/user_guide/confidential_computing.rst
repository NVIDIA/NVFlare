.. _confidential_computing:

#######################################################
Confidential Computing: Attestation Service Integration
#######################################################

Data used in NVFlare is encrypted during transmission between participants, which covers the communication between the NVFlare server, clients, and admin.  This security measure ensures
data in transit is well protected.  Users can also utilize existing infrastructure, such as storage encryption, to protect data at rest.  With confidential computing, NVFlare can protect data in use
and thus completes securing the entire lifecycle of data.

Confidential computing in NVFlare is designed to explicitly establish the trust between participants.  Each participant must first capture the evidence related to the hardware (such as a GPU), the software (GPU driver and VBIOS), and other components in its own platform.  The evidence will
be validated and signed to ensure its validity and authenticity.  The owner of signed evidence, called confidential computing token (CC token), can demonstrate the information about its computing environment to other
participants by providing the CC token. Upon receiving the CC token, the participant (the relying party) can verify the claims inside the CC token against its own security policy to determine whether the CC token owner is
using the required hardware, software, and components for security.  If the relying party finds the CC token does not meet its security policy, the relying party can inform the system that it chooses not to join the job deployment
and will not exchange models with others.  Only participants who trust and are trusted by one another will work together to run the NVFlare job.


**********************
Configuring CC Manager
**********************

In order to enable confidential computing in NVFlare, users need to include the CC manager, as a component, inside the resources.json file of startup kit local folder.  The entire NVFlare system must
be configured with the CC manager for either all participants or no participants.

The CC manager component depends on `NVIDIA Attestation SDK <https://github.com/NVIDIA/nvtrust/tree/main/guest_tools/attestation_sdk>`_.  Users have to install it as a prerequisite.  This SDK also
depends on other software stacks, such as GPU verifier, driver and others.

The following is the sample configuration of CC manager.

.. code-block:: json

    {
        "id": "cc_manager",
        "path": "nvflare.app_opt.confidential_computing.cc_manager.CCManager",
        "args": {
            "verifiers": [{"devices": "gpu", "env": "local", "url":"", "appraisal_policy_file":"evidence.plc","result_policy_file":"result.plc"}]
        }
    },


The ``id`` is used internally by NVFlare so that other components can get its instance.  The ``path`` is the complete Python module hierarchy.
The ``args`` contains only the verifiers, a list of possible verifiers.  Each verifier is a dictionary and its keys are "devices", "env", 
"url", "appraisal_policy_file" and "result_policy_file."


The value of devices is either "gpu" or "cpu" for current Attestation SDK.  The value of env is either "local" or "test" for the current Attestation SDK.
Currently, valid combination is gpu and local or cpu and test.  The value of url must be an empty string.
The appraisal_policy_file and result_policy_file must point to an existing file.  The former is currently ignored by Attestation SDK.
The latter currently supports the following content only

.. code-block:: json

    {
        "version":"1.0",
        "authorization-rules":{
            "x-nv-gpu-available":true,
            "x-nv-gpu-attestation-report-available":true,
            "x-nv-gpu-info-fetched":true,
            "x-nv-gpu-arch-check":true,
            "x-nv-gpu-root-cert-available":true,
            "x-nv-gpu-cert-chain-verified":true,
            "x-nv-gpu-ocsp-cert-chain-verified":true,
            "x-nv-gpu-ocsp-signature-verified":true,
            "x-nv-gpu-cert-ocsp-nonce-match":true,
            "x-nv-gpu-cert-check-complete":true,
            "x-nv-gpu-measurement-available":true,
            "x-nv-gpu-attestation-report-parsed":true,
            "x-nv-gpu-nonce-match":true,
            "x-nv-gpu-attestation-report-driver-version-match":true,
            "x-nv-gpu-attestation-report-vbios-version-match":true,
            "x-nv-gpu-attestation-report-verified":true,
            "x-nv-gpu-driver-rim-schema-fetched":true,
            "x-nv-gpu-driver-rim-schema-validated":true,
            "x-nv-gpu-driver-rim-cert-extracted":true,
            "x-nv-gpu-driver-rim-signature-verified":true,
            "x-nv-gpu-driver-rim-driver-measurements-available":true,
            "x-nv-gpu-driver-vbios-rim-fetched":true,
            "x-nv-gpu-vbios-rim-schema-validated":true,
            "x-nv-gpu-vbios-rim-cert-extracted":true,
            "x-nv-gpu-vbios-rim-signature-verified":true,
            "x-nv-gpu-vbios-rim-driver-measurements-available":true,
            "x-nv-gpu-vbios-index-conflict":true,
            "x-nv-gpu-measurements-match":true
        }
    }


****************
Runtime behavior
****************

When one participant, either server or client, starts, the CC manager reacts to EventType.SYSTEM_BOOTSTRAP and retrieves its own CC token via Attestation SDK after the Attestation SDK successfully communicates
with the software stacks and hardware.  This CC token will be stored locally in CC manager.

When the client registers itself with the server, it also includes its CC token in the registration data.  Server will collect the client's CC token if it successfully registers.  The server CC manager keeps
all client's CC tokens as well as its own token.

After a submitted job is scheduled to be deployed, the server verifies the CC tokens of clients that are included in the deployment map based on its result policy.  If server finds
 all tokens from clients in the deployment map are verified successfully, those tokens will be sent to clients in deployment map for client side verification.  The client can determine whether it
 wants to join this job or not based on the result of verifying others' CC tokens against its own result policy.  If one client decides not to join the job, server will not deploy that job to that client.

The server job scheduler will determine if the job has enough resources to be deployed and will determine the job's final status based on resource availability and retry policy.
