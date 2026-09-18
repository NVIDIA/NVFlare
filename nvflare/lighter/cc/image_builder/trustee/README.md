# Existing CoCo Trustee deployment

CVM Builder uses unmodified Trustee v0.22.0, paired with CoCo v0.23.0.
CoCo manages the running deployment. This directory contains reference
configuration and policy only; there are no systemd services, custom key servers,
Trustee source patches or guest-components patches.

Merge the relevant settings from [kbs.json](kbs.json) into the existing deployment
and follow [TRUSTEE_GUIDE.md](../TRUSTEE_GUIDE.md). Preserve existing workloads'
policies and references when selecting a compatible CVM security profile.

Vault builds upload through the native resource API using a scoped bearer token.
The policy-publisher role is separate from the resource role. KBS writes native
resources and releases them only after attestation and policy authorization.
Native uploads permit replacement; deletion has no permanent tombstone.
The CoCo operator owns upload fencing, credential rotation and backup recovery.
