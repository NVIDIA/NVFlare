# Provision Azure Confidential Computing participants

This example provisions an NVFlare server for Azure Confidential Container
Instances and a client for an Azure confidential VM. The `CCBuilder` installs
the Azure attestation authorizers and manager in both startup kits, applies the
restricted confidential-computing authorization policy, and enables startup
integrity signing.

Edit the participant names and Azure attestation settings in the four YAML
files, then run provisioning from this directory so the relative `cc_config`
paths resolve correctly:

```bash
cd examples/advanced/azure_cc
nvflare provision -p project.yml -w ./workspace
```

See the [unified Confidential Computing deployment guide](../../../docs/user_guide/confidential_computing/deployment.rst)
for the shared schema and the Azure resource, attestation, and launch boundary.
