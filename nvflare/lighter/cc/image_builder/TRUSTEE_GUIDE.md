# Use CoCo Trustee for CVM vault keys

CVM Builder uses **unmodified CoCo Trustee v0.22.0**, the Trustee release paired
with **CoCo v0.23.0**. There is no CVM Trustee fork or guest-components patch.
Use the existing Trustee deployment managed by CoCo. CVM Builder installs no
backend service, sidecar, or systemd unit. Its deployment inputs are Rego policies,
reference values, administrative role ACLs, and native resource uploads.

This guide describes the built-in AS/RVPS with upstream `local_fs` storage.
Vault builds upload through HTTPS and need no access to that storage. The offline
bundle-policy administration command still reads the policy namespace to verify
published bytes; other storage backends need an equivalent readback workflow.

## 1. Pin the upstream release

| Component | Pin |
|---|---|
| CoCo | v0.23.0 |
| Trustee | v0.22.0 / `512fed65642015b849f38fb13bfdec7806639987` |
| Guest-components used by Trustee's kbs-client | `da8d93f2797088a5f0636c8c1eeb31da73784fe8` (upstream Cargo.lock) |
| Policy engine | Regorus 0.11.0 |

Use the official KBS image at this release commit; upstream publishes commit
image tags rather than a `v0.22.0` image tag:

```sh
export TRUSTEE_IMAGE=ghcr.io/confidential-containers/staged-images/kbs:512fed65642015b849f38fb13bfdec7806639987
docker pull "$TRUSTEE_IMAGE"
docker image inspect "$TRUSTEE_IMAGE" --format '{{json .RepoDigests}}'
```

Record and deploy its immutable registry digest. For a native deployment, build
from the clean release checkout with upstream's documented build prerequisites
and Rust toolchain. Leave Cargo.toml, Cargo.lock, and guest-components unchanged:

```sh
git clone --branch v0.22.0 https://github.com/confidential-containers/trustee.git /tmp/trustee
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml   -p kbs --bin kbs --no-default-features --features coco-as-builtin
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml   -p kbs-client --bin kbs-client --features tdx-attester,snp-attester
./cvmctl provenance /tmp/trustee   /tmp/trustee/target/release/kbs /tmp/trustee_build.json
```

Keep the client's default crypto features. In this release, `native-tls` selects
an OpenSSL RSA decryptor incompatible with the builder's RSA-OAEP-256 resource
responses. Test encrypted key retrieval before packaging the client.

For the NVIDIA client feature, first follow [GPU_BUILD.md](GPU_BUILD.md) to build
NVAT from the revision in Trustee's Cargo.lock, apply the recorded Ubuntu 26.04
libxml2 compatibility patch, and install its header/library in the disposable
build environment. That guide then builds the client with `NVAT_USE_SYSTEM_LIB=1`
and `nvidia-attester`. The command above builds a CPU-only client. Compile guest
binaries for the guest's Linux environment.
The provenance command rejects dirty source and records the revision and binary
SHA-256; it does not build or modify Trustee.

Build CVM clients from this clean source checkout. Upstream's `sample_only`
kbs-client OCI artifacts are for sample-attester tests, not the TDX/SNP/NVIDIA
feature set needed here. A library or executable SHA-256 identifies file bytes;
an OCI manifest digest identifies a registry artifact. Do not interchange them.

## 2. Prepare storage and identities

Use CoCo's existing deployment manifests, service account, storage volumes, TLS
endpoint and lifecycle management. Protect the host against swap and core dumps.
The illustrative paths below are inside that deployment; adapt them to its
mounts rather than installing another Trustee instance.

KBS needs write access to `storage/repository` for native resource upload/delete
and `storage/kbs` for resource policy updates. Keep AS policies and endorsed
references controlled by the deployment operator. Apply network restrictions,
process limits and secret handling through CoCo's deployment configuration.

## 3. Create certificates and administrative identities

Use your organization's PKI for production. For a first test, this complete
example creates a local CA, HTTPS server certificate, AS signing certificate,
and an Ed25519 KBS administrative signing key.
The AS key is P-256 because the CVM profile validates ES256 tokens.

```sh
umask 077
export TRUSTEE_PKI="$HOME/workspace/cvm-trustee-pki"
export CVM_BUILDER_SOURCE="$PWD"
export TRUSTEE_DNS=kbs.example.org
mkdir -p "$TRUSTEE_PKI"
cd "$TRUSTEE_PKI"
for authority in ca as-ca; do
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "$authority.key" -out "$authority.pem" -days 365 \
    -subj "/CN=CVM Trustee $authority" \
    -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
    -addext 'keyUsage=critical,keyCertSign,cRLSign'
done
for identity in server as; do
  openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "$identity.key" -out "$identity.csr" -subj "/CN=$identity"
  printf '%s\n' 'basicConstraints=critical,CA:FALSE' \
    'keyUsage=critical,digitalSignature' > "$identity.ext"
  if [ "$identity" = server ]; then
    printf '%s\n' 'extendedKeyUsage=serverAuth' \
      "subjectAltName=DNS:$TRUSTEE_DNS,DNS:localhost,IP:127.0.0.1,IP:10.0.2.2" >> "$identity.ext"
  else
    printf '%s\n' 'extendedKeyUsage=clientAuth' >> "$identity.ext"
  fi
  issuer=ca
  if [ "$identity" = as ]; then issuer=as-ca; fi
  openssl x509 -req -in "$identity.csr" -CA "$issuer.pem" -CAkey "$issuer.key" \
    -CAcreateserial -out "$identity.pem" -days 90 -sha256 -extfile "$identity.ext"
done
cat as.pem as-ca.pem > as-chain.pem
openssl pkey -in as.key -pubout -out as-public.pem
openssl genpkey -algorithm Ed25519 -out kbs-admin.key
openssl pkey -in kbs-admin.key -pubout -out kbs-admin.pub
# Install these files using the existing CoCo deployment's secret-management workflow.
cd "$CVM_BUILDER_SOURCE"
```

The AS signing root must issue only AS signing identities. Never add the transport
CA to KBS `trusted_certs_paths`: a builder client certificate must not be able to
sign an accepted attestation token. Test direct resource retrieval with EARs
signed by builder, admin, server, and an unrelated signer; every request must fail.

Retain `ca.key`, `as-ca.key`, and `kbs-admin.key` with the deployment operator.
Give builders only the HTTPS CA and a pre-issued resource-administration bearer
token (§5). Public `as-public.pem` and `ca.pem` go into CVM profile inputs.
No backend private key or administrative token belongs in a CVM, application
vault, or OCI delivery.

## 4. Configure upstream KBS

Merge the relevant settings from [trustee/kbs.json](trustee/kbs.json) into the
existing CoCo KBS configuration; adjust listener and certificate paths. This is the upstream v0.22 configuration
schema. Its `cvm-policy` role can publish resource policy and query named
references. It cannot replace AS policies, register references, or mutate native
KBS resources. The separate `cvm-resources` role permits native resource POST and DELETE under
`keys/`; restrict its path expression to the approved bundle prefixes when
issuing deployment-specific credentials. It cannot publish policy. This upstream
ACL is endpoint-based: do not describe its resource token as create-only.

The reference configuration explicitly selects Intel's `standard` TCB update
channel for DCAP verification:

```json
"verifier_config": {
  "dcap_verifier": {
    "collateral_service": "https://api.trustedservices.intel.com/sgx/certification/v4/",
    "use_secure_cert": true,
    "tcb_update_type": "standard"
  }
}
```

Trustee v0.22.0 defaults to `early` when this setting is omitted. It fetches
collateral itself, so setting the host's `/etc/sgx_default_qcnl.conf` alone does
not configure this verifier. `standard` retains Intel's mitigation deployment
grace period; `early` applies newer TCB recovery requirements. The same signed
quote can therefore pass under `standard` and report `OutOfDate` under `early`.
See [Intel's TCB recovery guidance](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/best-practices/trusted-computing-base-recovery.html).

Select and record the channel as part of the deployment's security policy.
Deployments requiring the early baseline should set `tcb_update_type` to
`early` and update platform firmware accordingly. The CPU appraisal policy
requires upstream's `UpToDate` status and unexpired collateral under the selected
channel; it does not accept `OutOfDate` as a fallback.

For diagnosis and verified recovery steps, see
[Intel TDX troubleshooting](TDX_TROUBLESHOOTING.md#tcb-channel-selection),
including the distinction between QGS quote-generation failures and a valid
quote appraised against a newer TCB baseline.

The upstream `kbs-client` CLI uses the `default` AS policy. Set
`attestation_policy_id: default` and
`token_issuer: CoCo-Attestation-Service` in the CVM profile. Install the reviewed
CPU policy as `storage/attestation_service_policy/default_cpu.rego`; install a GPU
profile's generated `gpu_attestation_policy.rego` as `default_gpu.rego`.
Policy content hashes and the profile version identify the approved policy
revision. A policy name by itself is not approval.

Each CPU/GPU security profile keeps separate policy/reference storage and its
own configured endpoint. Reuse the same upstream Trustee image for these
instances. Do not overwrite an existing CoCo deployment's default policies or
combine different profiles' TCB allowlists. The upstream protocol also supports
policy selectors, but this release's `kbs-client` CLI does not expose one.

Before starting KBS, populate all four default policy files so its startup can
find them with its policy namespace mounted read-only:

```sh
sudo install -m 0600 -o cvm-trustee -g cvm-trustee config/attestation_policy.rego   /var/lib/cvm-trustee/storage/attestation_service_policy/default_cpu.rego
python3 - <<'PY_POLICY'
from pathlib import Path
policy = 'package policy\nimport rego.v1\ntrust_claims := {"hardware": 97}\n'
Path('/tmp/cvm_deny_device.rego').write_text(policy)
PY_POLICY
for device in gpu switch ppcie; do
  sudo install -m 0600 -o cvm-trustee -g cvm-trustee /tmp/cvm_deny_device.rego     "/var/lib/cvm-trustee/storage/attestation_service_policy/default_${device}.rego"
done
```

On a new deployment, also install the initial deny-all resource policy:

```sh
sudo install -m 0600 -o cvm-trustee -g cvm-trustee trustee/resource_policy.rego \
  /var/lib/cvm-trustee/storage/kbs/resource-policy.rego
```

Do not reset an existing deployment's published policy. Require this file to
exist at startup, before accepting traffic; apply this check in the existing CoCo
Kubernetes init container. Upstream's built-in fallback resource policy is not
the CVM authorization policy and must not be used when a policy volume is missing.

For a GPU profile replace `default_gpu.rego` with that bundle's generated GPU
policy. Add `nvidia_verifier` to the existing `verifier_config` object under
`attestation_service` in `kbs.json`, preserving the DCAP settings:

```json
"verifier_config": {
  "dcap_verifier": {
    "collateral_service": "https://api.trustedservices.intel.com/sgx/certification/v4/",
    "use_secure_cert": true,
    "tcb_update_type": "standard"
  },
  "nvidia_verifier": {
    "type": "Remote",
    "verifier_url": "https://nras.attestation.nvidia.com/v4/attest"
  }
}
```

Trustee appends `/gpu` when calling NRAS. Set the profile's
`gpu_attestation_url` to the resulting `/v4/attest/gpu` URL for its deployment
contract and backend fault tests. Permit the backend to reach NRAS and its JWKS
endpoint, plus required Intel/AMD collateral services. NVIDIA signatures and
nonce verification use CoCo's upstream implementation and trust handling; there
is no separate CVM JWKS snapshot or verifier configuration environment variable.
Use upstream's SNP offline certificate store when KDS outage tolerance is required.

The GPU Rego policy requires successful NRAS report/RIM checks, secure boot,
disabled debug, and approved driver/VBIOS versions. It uses only upstream claims;
it does not rely on custom `verifier` or `arch` marker fields. Optional version,
device-type and schema fields must agree when present. The KBS resource policy
requires fresh CPU/GPU appraisals, exact device count and distinct GPU identities.

## 5. Use native Trustee resource administration

The trusted token issuer creates a signed JWT accepted by the existing KBS admin
identity provider. For the example ACL, use `role: cvm-resources`,
`iss: cvm-builder`, `aud: coco-trustee`, and valid `iat`, `nbf`, and `exp` claims.
Set its validity to cover the build and rotate it through your existing credential
workflow. Keep the issuer's signing key off build workers; possession of that key
would allow minting tokens for other roles.

Configure the build worker's `cvm_project.yml`:

```yaml
trustee:
  url: https://kbs.example.org:8443
  ca: ./credentials/kbs-ca.pem
  admin_token_file: ./credentials/kbs-resource-token.jwt
```

The token file contains only the signed JWT, with restricted file permissions.
Paths resolve relative to this YAML. Stage 2 validates the vault, then uploads its
64-byte key with `POST /kbs/v0/resource/keys/<build_id>/<binding_id>` using that
bearer token. Guest retrieval still requires successful attestation and resource
policy authorization; administration credentials are never delivered to guests.

Native POST permits replacement. Builds always seal fresh vaults and issue one
upload attempt; they do not automatically retry an uncertain request. Preserve
`provisioning.json` and `build_failure.json` for operator recovery. Native DELETE
has no permanent tombstone, so operators must fence uploads before revocation and
preserve deletions when restoring backups. These are the existing CoCo lifecycle
responsibilities; no CVM key adapter or reconciliation daemon is required.

## 6. Install references and policies

Stop the relevant KBS instance while changing its approved AS policies or RVPS
references. Import a finalized bundle's reviewed values with an operator-chosen
expiry:

```sh
sudo -u cvm-trustee ./cvmctl references /path/to/bundle   --store /var/lib/cvm-trustee/storage/reference_value   --state /var/lib/cvm-trustee/admin --expires 2026-12-01T00:00:00Z
```

The import uses upstream RVPS record files and a `cvm_reference_expiry` companion
reference. The AS policies call `query_reference_value()` and enforce each
record's deadline: v0.22 RVPS does not itself reject an expired record. Existing
approvals cannot be silently broadened or renewed by this importer. Missing or
expired approvals deny appraisal. Measurements remain in Trustee property storage.

Apply configuration through CoCo's normal deployment mechanism. Run
`./cvmctl preflight trustee` on the trusted deployment host and confirm the
initial deny-all resource policy exists before accepting traffic. The repository
contains no Trustee systemd services to install or enable.

Use a fresh profile version and rebuild/reapprove bundles when migrating from
the earlier backend. Its files, policies, receipts, and guest binaries are not
wire- or storage-compatible with v0.22. Existing hardware acceptance does not
approve the new profile.

## 7. Verify the deployment and enable a bundle

Administration runs beside the same local_fs storage. Configure `admin.json`:

```json
{
  "url": "https://kbs.example.org:8443",
  "ca": "/etc/cvm-trustee/pki/ca.pem",
  "admin_private_key": "/protected/kbs-admin.key",
  "admin_role": "cvm-policy",
  "admin_issuer": "cvm-builder",
  "admin_audience": "coco-trustee",
  "storage_directory": "/var/lib/cvm-trustee/storage",
  "state": "/var/lib/cvm-trustee/admin",
  "trustee_binary": "/opt/cvm-trustee/bin/kbs",
  "trustee_build": "/etc/cvm-trustee/trustee_build.json",
  "deployment_receipt": "/etc/cvm-trustee/deployment_receipt.json"
}
```

Copy the upstream build provenance to `trustee_build.json`. For image deployments,
retain the registry digest and source-commit provenance alongside the extracted
binary digest. The deployment receipt contains:

```json
{
  "trustee_commit": "512fed65642015b849f38fb13bfdec7806639987",
  "source_clean": true,
  "policy_selection_tested": true,
  "unauthorized_administration_denied": true,
  "immutable_as_policies": {
    "default_cpu": "SHA256_OF_INSTALLED_CPU_POLICY",
    "default_gpu": "SHA256_OF_INSTALLED_GPU_POLICY_FOR_GPU_PROFILES"
  }
}
```

Set acceptance fields only after testing the actual deployment. Verify denied
AS-policy replacement and native resource writes even with the policy publisher's
credentials; builder/admin/transport certificates cannot sign accepted EARs;
wrong, stale, cross-vault or incomplete GPU appraisals deny release; key retry,
revoke and restore behavior remains correct. Verify the measured guest and KBS
emit and accept the expected default CPU/GPU policy. Complete the bundle's
hardware acceptance before production approval.

```sh
sudo ./cvmctl admin install admin.json /path/to/approved/bundle
```

`GET /kbs/v0/resource-policy` now returns policy IDs, not encoded policy bytes.
Administration checks that listing and verifies the exact bytes in the shared
`kbs/resource-policy.rego` file. Named references are queried at
`GET /kbs/v0/reference-value/<name>`. Do not reuse old response decoders.

Configure the vault builder's scoped resource token in `cvm_project.yml` and
run Vault Build normally. Only key creation occurs per vault; measurements and
resource-policy rules are registered per generic bundle.

## 8. Revoke, retire and restore

First stop or fence builds targeting the resource. With a resource-role config
containing `url`, `ca` and `admin_token_file`, revoke a vault through native KBS:

```sh
python3 -m cvm admin revoke resources.json keys/BUILD_ID/BINDING_ID
```

This sends `DELETE /kbs/v0/resource/keys/<build_id>/<binding_id>`. It prevents
future retrieval while the resource is absent; an authorized POST can recreate
it. Token revocation/expiry and backup recovery belong to the CoCo operator.

Retire a generic bundle with `./cvmctl admin retire admin.json <build_id>`.
This removes its reusable resource rule and records retirement in the publisher's
administrative state. Keys may remain stored, but the retired rule no longer
permits release. Preserve the current policy and retirement state during restore;
verify denials before reopening traffic. An unsuccessful policy update is not a
completed retirement: retry the command and verify readback. Deletion and policy
changes cannot retract secrets already released to a running guest.

If an earlier deployment supplies `key_service_state`, both `admin install` and
`admin retire` reject the configuration, even when the value is empty or null.
Before removing that field, fence builds, policy publication and key release;
apply outstanding legacy revocations through CoCo Trustee and preserve them in
the operator's recovery records. Merge every legacy `retired/` marker into the
publisher's configured `state/retired/`, preserving existing markers, bundle IDs
and the current resource policy. Do not discard markers for bundles that are
currently absent. Verify the migrated state before removing `key_service_state`,
then verify retired bundles and revoked resources remain denied before reopening
traffic. Updating the builder does not migrate this state automatically.

## Upstream contracts

- [CoCo release pairing](https://github.com/confidential-containers/trustee/releases/tag/v0.22.0)
- [KBS configuration and ACLs](https://github.com/confidential-containers/trustee/blob/v0.22.0/kbs/docs/config.md)
- [Storage namespaces](https://github.com/confidential-containers/trustee/blob/v0.22.0/deps/key-value-storage/README.md)
- [NVIDIA verifier](https://github.com/confidential-containers/trustee/blob/v0.22.0/deps/verifier/src/nvidia/README.md)
- [SNP offline certificates](https://github.com/confidential-containers/trustee/blob/v0.22.0/attestation-service/docs/amd-offline-certificate-cache.md)
