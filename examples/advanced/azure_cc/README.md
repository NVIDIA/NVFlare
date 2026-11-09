# Provision an Azure confidential server

This example uses one Azure Container Instances (ACI) server and a confidential
virtual-machine client. ACI attestation is signed evidence about the server's
protected environment. Provisioning configures every ACI checker to require the
fingerprint of the administrator-approved enforcement policy.

The server image stays fixed. Its startup program verifies a separately signed
startup kit, the generated configuration and credentials, supplied by the
administrator through a read-only Azure secret volume at launch. Do not copy
the final startup kit or policy fingerprint into the image: doing so changes the
image for which the policy was prepared.

Follow the [Azure deployment guide](../../../docs/user_guide/confidential_computing/azure/confidential_azure_container_instances_deployment.rst)
for reproducible commands in this order:

1. Create the deployment-signing private and public keys. Keep the private key
   local; the public key goes into the fixed server image.
2. Put the approved NVFlare installation wheel in `wheels/`, then build and push
   the supplied `Dockerfile`. The `.dockerignore` excludes private material.
3. Prepare the Azure template, including the fixed image and startup command and
   the read-only mount at `/run/nvflare-input`. Generate the enforcement policy
   and decode it to `approved-cce-policy.txt` without changing its contents.
4. Edit participant names and attestation settings. The server's `cc_server.yml`
   requires `cce_policy_file`, the raw decoded policy path, and
   `startup_kit_signing_key_file`, the unencrypted private deployment key path.
   Relative paths resolve against that configuration file. The example expects
   `deployment-private-key.pem` beside it; no key or policy is supplied here.
5. Run provisioning from this directory:

   ```bash
   nvflare provision -p project.yml -w ./workspace
   ```

6. In the production output's server directory, use `startup-kit.tar`, the
   uncompressed signed kit archive; `startup-kit.sig`, its separate binary
   signature; and `approved_cce_policy.txt`, the snapshotted raw policy.
   The delivery manifest lists these files and their checksums, calculated
   identifiers used to detect changes. Encode each file once for its Azure
   transport field, following the guide. Keep that launch document private.
7. Directly launch ACI with the same prepared image and policy. Do not regenerate
   policy or rebuild the image after provisioning. Distribute the client kit
   through the usual confidential-machine deployment process.

Provisioning is the approval step and does not ask for confirmation. Missing or
different policy fingerprints are rejected; existing ACI deployments must be
reprovisioned. One server-wide fingerprint is used without a participant-specific
map. Confidential-Containers clients can use the corresponding configuration in
the [unified deployment guide](../../../docs/user_guide/confidential_computing/deployment.rst).

This direct-launch workflow trusts the administrator and Azure deployment
service with the kit credentials. Signatures provide integrity, not encrypted
delivery or attestation-controlled key release. Previously signed archives remain
acceptable. Rotating the deployment key requires a new image, policy, and
provisioning run; changing only the policy still requires reprovisioning and
redistribution of the affected kits.

After launch, perform the guide's administrator-run smoke checks: successful
registration with the approved policy, rejection of fresh evidence for a different
policy, and startup refusal for a damaged archive or signature. Provisioning
itself does not deploy Azure resources.
