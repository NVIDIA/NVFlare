# Existing CoCo Trustee deployment

CVM Builder uses unmodified Trustee v0.22.0, paired with CoCo v0.23.0.
CoCo manages the running deployment. This directory contains reference
configuration and policy only; there are no systemd services, custom key servers,
Trustee source patches or guest-components patches.

Merge the relevant settings from [kbs.json](kbs.json) into the existing deployment
and follow [TRUSTEE_GUIDE.md](../TRUSTEE_GUIDE.md). Preserve existing workloads'
policies and references when selecting a compatible CVM security profile.

The reference configuration explicitly uses Intel's `standard` TCB update
channel. Upstream Trustee defaults to `early`; choose the channel as a deployment
security policy and keep it explicit. CPU appraisal still requires `UpToDate`
and unexpired collateral. See the guide for the firmware baseline implications.

Vault builds upload through the native resource API using a scoped bearer token
valid for at most 30 days. The policy-publisher role is separate from the
resource role, and the sample ACL shows a bundle-scoped `cvm-resources-<build_id>`
role beside the generic one; print a bundle's entry with `cvmctl admin acl`. KBS
writes native resources and releases them only after attestation and policy
authorization. Native uploads permit replacement; deletion has no permanent
tombstone. The CoCo operator owns upload fencing, credential rotation and backup
recovery. Bundle installation requires a receipt signed by an acceptance
authority listed in the administration configuration.
