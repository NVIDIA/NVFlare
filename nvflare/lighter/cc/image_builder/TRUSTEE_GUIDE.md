# Use CoCo Trustee for CVM vault keys

CVM Builder uses **unmodified CoCo Trustee v0.22.0**, the Trustee release paired
with **CoCo v0.23.0**. There is no CVM Trustee fork or guest-components patch.
Deploy the same upstream distribution and image digest used by your CoCo
installation. CVM-specific requirements are Rego policies, reference values,
role ACLs, and the vault key provisioning adapter.

This guide describes the built-in AS/RVPS with the upstream `local_fs` backend.
The builder's administration and create-only key adapter run beside that storage,
including as sidecars with the corresponding shared volumes. They do not replace
KBS, AS, RVPS, or NVIDIA verification. Other CoCo storage backends require a
compatible provisioning adapter; do not point this filesystem adapter at Redis,
Vault, or a remote filesystem layout.

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
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml   -p kbs-client --bin kbs-client --no-default-features   --features native-tls,tdx-attester,snp-attester,nvidia-attester
python3 scripts/trustee_provenance.py /tmp/trustee   /tmp/trustee/target/release/kbs /tmp/trustee_build.json
```

The NVIDIA client feature uses upstream's NVAT SDK bindings and requires the
matching `libnvat` development/runtime libraries. CPU-only clients may omit
`nvidia-attester`. Compile guest binaries for the guest's Linux environment.
The provenance command rejects dirty source and records the revision and binary
SHA-256; it does not build or modify Trustee.

## 2. Prepare storage and identities

Run the backend on a trusted host with swap disabled and piped core collectors
disabled. The systemd templates expect the builder under `/opt/cvm-builder`, its
Python requirements installed, and the upstream binary at `/opt/cvm-trustee/bin/kbs`.
For Kubernetes, apply equivalent volume permissions, process limits, and network
policy to the upstream CoCo deployment.

```sh
export CVM_BUILDER_SOURCE="$PWD"
export TRUSTEE_DNS=kbs.example.org
sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin cvm-trustee
sudo install -d -m 0750 -o root -g cvm-trustee /etc/cvm-trustee /etc/cvm-trustee/pki
sudo install -d -m 0700 -o cvm-trustee -g cvm-trustee   /var/lib/cvm-trustee/storage /var/lib/cvm-trustee/storage/kbs   /var/lib/cvm-trustee/storage/repository   /var/lib/cvm-trustee/storage/attestation_service_policy   /var/lib/cvm-trustee/storage/reference_value   /var/lib/cvm-trustee/admin /var/lib/cvm-trustee-revocations
```

## 3. Create certificates and administrative identities

Use your organization's PKI for production. For a first test, this complete
example creates a local CA, HTTPS server certificate, AS signing certificate,
builder/admin client certificates, and an Ed25519 KBS administrative key.
The AS key is P-256 because the CVM profile validates ES256 tokens.

```sh
umask 077
export TRUSTEE_PKI="$HOME/workspace/cvm-trustee-pki"
mkdir -p "$TRUSTEE_PKI"
cd "$TRUSTEE_PKI"
for authority in ca as-ca; do
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "$authority.key" -out "$authority.pem" -days 365 \
    -subj "/CN=CVM Trustee $authority" \
    -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
    -addext 'keyUsage=critical,keyCertSign,cRLSign'
done
for identity in server as builder admin; do
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
sudo install -m 0640 -o root -g cvm-trustee \
  ca.pem as-ca.pem server.pem server.key as.key as-chain.pem kbs-admin.pub /etc/cvm-trustee/pki/
cd "$CVM_BUILDER_SOURCE"
```

The AS signing root must issue only AS signing identities. Never add the transport
CA to KBS `trusted_certs_paths`: a builder client certificate must not be able to
sign an accepted attestation token. Test direct resource retrieval with EARs
signed by builder, admin, server, and an unrelated signer; every request must fail.

Retain `ca.key`, `as-ca.key`, and `kbs-admin.key` with the administrator. Give only `builder.pem`,
`builder.key`, and `ca.pem` to the vault builder. The key-service admin client uses
`admin.pem`/`admin.key`; it is distinct from the Ed25519 KBS policy administrator.
Public `as-public.pem` and `ca.pem` go into CVM profile inputs. No backend private
key belongs in a CVM, an application vault, or an OCI delivery.

## 4. Configure upstream KBS

Install [trustee/kbs.json](trustee/kbs.json) at `/etc/cvm-trustee/kbs.json` and
adjust listener and certificate paths. This is the upstream v0.22 configuration
schema. Its `cvm-policy` role can publish resource policy and query named
references. It cannot replace AS policies, register references, or mutate native
KBS resources. Do not add a broad administrator ACL for this identity.

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
exist at startup, as the supplied systemd unit does; apply the same check in a
Kubernetes init container. Upstream's built-in fallback resource policy is not
the CVM authorization policy and must not be used when a policy volume is missing.

For a GPU profile replace `default_gpu.rego` with that bundle's generated GPU
policy. Add the following object under `attestation_service` in `kbs.json`:

```json
"verifier_config": {
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

## 5. Provision vault keys

`builder.key_service` is a CVM application adapter for atomic create-only uploads
and persistent revocation. Native KBS resource writes overwrite existing keys,
so the adapter retains the vault lifecycle semantics without patching KBS.
Use this configuration as `/etc/cvm-trustee/key-service.json`:

```json
{
  "listen": "0.0.0.0",
  "port": 9443,
  "resources": "/var/lib/cvm-trustee/storage/repository",
  "state": "/var/lib/cvm-trustee-revocations",
  "client_ca": "/etc/cvm-trustee/pki/ca.pem",
  "cert": "/etc/cvm-trustee/pki/server.pem",
  "key": "/etc/cvm-trustee/pki/server.key",
  "certificate_roles": {
    "BUILDER_CERTIFICATE_DER_SHA256": "builder",
    "ADMIN_CERTIFICATE_DER_SHA256": "admin"
  }
}
```

Replace fingerprints using SHA-256 of each client certificate's DER encoding.
Initialize `approved-bundles.json` in the revocation state to `{"build_ids":[]}`.
The adapter stores `keys/build/binding` as the upstream local_fs filename
`keys\x2Fbuild\x2Fbinding` inside the `repository` namespace. KBS mounts this
namespace read-only; the adapter writes it. Keep revocation state separate from
resource backups. An identical upload is an idempotent retry; different bytes at
the same path conflict. Only the adapter's administrator can revoke a key.

## 6. Install references and policies

Stop the relevant KBS instance while changing its approved AS policies or RVPS
references. Import a finalized bundle's reviewed values with an operator-chosen
expiry:

```sh
sudo -u cvm-trustee python3 scripts/trustee_references.py /path/to/bundle   --store /var/lib/cvm-trustee/storage/reference_value   --state /var/lib/cvm-trustee/admin --expires 2026-12-01T00:00:00Z
```

The import uses upstream RVPS record files and a `cvm_reference_expiry` companion
reference. The AS policies call `query_reference_value()` and enforce each
record's deadline: v0.22 RVPS does not itself reject an expired record. Existing
approvals cannot be silently broadened or renewed by this importer. Missing or
expired approvals deny appraisal. Measurements remain in Trustee property storage.

Install the systemd templates from `trustee/systemd/`, or equivalent Kubernetes
services. Configure the key service with locked memory (`LimitMEMLOCK=infinity`),
zero core limit, and a restrictive umask. The supplied KBS unit mounts resource,
AS-policy and reference namespaces read-only while allowing the resource-policy
namespace and DCAP cache to be written. Run reconciliation before KBS starts.

```sh
sudo install -m 0644 trustee/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start cvm_key_service.service cvm_trustee_kbs.service
```

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
  "resources": "/var/lib/cvm-trustee/storage/repository",
  "key_service_state": "/var/lib/cvm-trustee-revocations",
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
sudo scripts/admin_install admin.json /path/to/approved/bundle
```

`GET /kbs/v0/resource-policy` now returns policy IDs, not encoded policy bytes.
Administration checks that listing and verifies the exact bytes in the shared
`kbs/resource-policy.rego` file. Named references are queried at
`GET /kbs/v0/reference-value/<name>`. Do not reuse old response decoders.

Configure the vault builder's mTLS upload credentials in `cvm_project.yml` and
run Vault Build normally. Only key creation occurs per vault; measurements and
resource-policy rules are registered per generic bundle.

## 8. Revoke, retire and restore

Revoke a vault with an authenticated key-service administrator's DELETE to
`/v1/resources/keys/<build_id>/<binding_id>`. Retire a generic bundle with
`scripts/admin_retire admin.json <build_id>`; this removes its resource rule and
all its keys. Never roll permanent revocation state back with a resource backup.
Stop KBS before restoring, run `builder.key_service --reconcile` with its config,
then verify revoked keys remain denied before starting KBS.

## Upstream contracts

- [CoCo release pairing](https://github.com/confidential-containers/trustee/releases/tag/v0.22.0)
- [KBS configuration and ACLs](https://github.com/confidential-containers/trustee/blob/v0.22.0/kbs/docs/config.md)
- [Storage namespaces](https://github.com/confidential-containers/trustee/blob/v0.22.0/deps/key-value-storage/README.md)
- [NVIDIA verifier](https://github.com/confidential-containers/trustee/blob/v0.22.0/deps/verifier/src/nvidia/README.md)
- [SNP offline certificates](https://github.com/confidential-containers/trustee/blob/v0.22.0/attestation-service/docs/amd-offline-certificate-cache.md)
