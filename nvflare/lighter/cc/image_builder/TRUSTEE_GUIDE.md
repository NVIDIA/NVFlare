# Deploy upstream Trustee for CVM vault keys

CVM Builder uses the **upstream CoCo Trustee v0.22.0 service**, the Trustee
release paired with **CoCo v0.23.0**. There is no CVM Trustee server fork. A
reviewed client-only patch exposes guest-components' upstream attestation-policy
selector through the standalone `kbs-client` CLI. You may share the Trustee
instance already managed by CoCo or deploy the same upstream KBS as a standalone
service for `bare_metal_cvm`. CVM Builder installs no backend service, sidecar,
or systemd unit. The operator owns the service manager or container deployment;
the repository supplies a complete base KBS configuration, policies, reference
import, provenance/receipt tooling, and native administration clients.

This guide describes the built-in AS/RVPS with upstream `local_fs` storage.
Vault builds upload through HTTPS and need no access to that storage. The offline
bundle-policy administration command still reads the policy namespace to verify
published bytes; other storage backends need an equivalent readback workflow.

Deploy one independently administered Trustee instance and storage/admin state
per project or tenant. That instance can serve CoCo plus multiple isolated CVM
security profiles. The supplied endpoint ACLs and bundle-scoped resource roles
are not a multi-tenant isolation boundary. All holders must belong to the same
trusted administration domain. For mutually untrusted tenants, provision
separate instances and credentials.

## 1. Pin the upstream release

| Component | Pin |
|---|---|
| CoCo | v0.23.0 |
| Trustee | v0.22.0 / `512fed65642015b849f38fb13bfdec7806639987` |
| Guest-components used by CVM kbs-client | `706cd996d2f3ef746c3ec378741d6d953b08740f` (upstream selector API) |
| Policy engine | Regorus 0.11.0 |

Use the release commit in every shared CoCo/CVM service. The CoCo deployment
example consumes an existing Trustee; it does not build or install the service.
The deployment operator maintains the server build/provenance records separately.
An official image can be used when its immutable digest and extracted server
binary are recorded:

```sh
export TRUSTEE_IMAGE=ghcr.io/confidential-containers/staged-images/kbs:512fed65642015b849f38fb13bfdec7806639987
docker pull "$TRUSTEE_IMAGE"
docker image inspect "$TRUSTEE_IMAGE" --format '{{json .RepoDigests}}'
```

Record and deploy its immutable registry digest. For a clean native deployment,
build with upstream's documented prerequisites and Rust toolchain. Build and
record the service before applying the client patch:

```sh
git clone --branch v0.22.0 https://github.com/confidential-containers/trustee.git /tmp/trustee
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml   -p kbs --bin kbs --no-default-features --features coco-as-builtin
./cvmctl provenance /tmp/trustee   /tmp/trustee/target/release/kbs /tmp/trustee_build.json
git -C /tmp/trustee apply "$PWD/cvm/build/kbs_client_policy_selector.patch"
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml \
  -p kbs-client --bin kbs-client --features tdx-attester,snp-attester
./cvmctl provenance /tmp/trustee \
  /tmp/trustee/target/release/kbs-client /tmp/kbs_client_cpu_build.json
```

Keep the client's default crypto features. In this release, `native-tls` selects
an OpenSSL RSA decryptor incompatible with the builder's RSA-OAEP-256 resource
responses. Test encrypted key retrieval before packaging the client.

For the NVIDIA client feature, first follow [GPU_BUILD.md](GPU_BUILD.md) to build
NVAT from the clean 2026.06.09 source revision pinned by the selector-capable
client lockfile and install its header/library in the disposable build
environment. That revision already contains the Ubuntu 26.04 `libxml2`
const-correctness fix, so no local NVAT patch is applied. The guide then builds
the client with `NVAT_USE_SYSTEM_LIB=1` and `nvidia-attester`. The command above
builds a CPU-only client. Compile guest binaries for the guest's Linux
environment.
The provenance command accepts either the clean server checkout or exactly the
checked-in client patch. The CoCo deployment operator records the service's reviewed
build recipe, source-patch digest, dependency hashes, image ID and exported
server-binary digest. Any other source change or recipe is rejected.

Build CVM clients from this clean source checkout. Upstream's `sample_only`
kbs-client OCI artifacts are for sample-attester tests, not the TDX/SNP/NVIDIA
feature set needed here. A library or executable SHA-256 identifies file bytes;
an OCI manifest digest identifies a registry artifact. Do not interchange them.

## 2. Choose a deployment path

Both paths use the same pinned KBS binary and schema:

- **Shared CoCo Trustee:** keep CoCo's deployment manifests, service account,
  volumes, TLS endpoint, AS signer, and lifecycle management. Merge the CVM
  selectors, ACLs, policies, and references described below without replacing
  CoCo's default policy or records.
- **Standalone upstream Trustee:** create a dedicated service identity and
  persistent local filesystem, install the clean KBS binary built in §1, and
  use [trustee/kbs.json](trustee/kbs.json) as the base schema template. It is
  intentionally not deployable unchanged: replace its sample selectors and
  bundle-scoped ACL, add the required SNP/GPU verifier entries described in §5,
  and install only reviewed policies and references. Create the unprivileged
  service identity and install the
  binary, provenance, configuration, and initial deny-all resource policy:

  ```bash
  getent passwd cvm-trustee >/dev/null || \
    sudo useradd --system --home-dir /var/lib/cvm-trustee \
      --shell /usr/sbin/nologin cvm-trustee
  sudo install -d -m 0755 /opt/cvm-trustee/bin
  sudo install -m 0755 /tmp/trustee/target/release/kbs /opt/cvm-trustee/bin/kbs
  sudo install -d -m 0700 -o cvm-trustee -g cvm-trustee \
    /var/lib/cvm-trustee/storage/repository \
    /var/lib/cvm-trustee/storage/kbs \
    /var/lib/cvm-trustee/storage/reference_value \
    /var/lib/cvm-trustee/storage/attestation_service_policy \
    /var/lib/cvm-trustee/admin
  sudo install -d -m 0750 -o root -g cvm-trustee \
    /etc/cvm-trustee /etc/cvm-trustee/pki
  sudo install -m 0640 -o root -g cvm-trustee trustee/kbs.json /etc/cvm-trustee/kbs.json
  sudo install -m 0640 -o root -g cvm-trustee /tmp/trustee_build.json \
    /etc/cvm-trustee/trustee-build.json
  sudo install -m 0600 -o cvm-trustee -g cvm-trustee trustee/resource_policy.rego \
    /var/lib/cvm-trustee/storage/kbs/resource-policy.rego
  ```

  Generate and install the PKI in §4, replace the sample DNS name and policy
  IDs, and install the reviewed AS policies and references in §§5 and 7. Then
  run KBS under a supervised service as `cvm-trustee`, with restart limits,
  private logs, no swap/core dumps, and only the required network access. This
  foreground command is a final smoke test, not a production process
  supervisor:

  ```sh
  sudo -u cvm-trustee /opt/cvm-trustee/bin/kbs \
    --config-file /etc/cvm-trustee/kbs.json
  ```

The checked-in JSON selects persistent `local_fs` storage and built-in CoCo AS
and RVPS. It is the standalone base configuration; a CoCo operator merges only
the relevant fields into that deployment's native JSON or TOML. Do not start
with an empty KBS config: omitted storage becomes in-memory and omitted CVM
resource policy falls back to a policy that does not express this design.

## 3. Prepare storage and identities

For a CoCo deployment, use its existing service account, storage volumes, TLS
endpoint and lifecycle management. For a standalone deployment, create the
paths in §2 and map the same paths into the KBS container or native process.
Protect either host against swap and core dumps.

KBS needs write access to `storage/repository` for native resource upload/delete
and `storage/kbs` for resource policy updates. Keep AS policies and endorsed
references controlled by the deployment operator. Apply network restrictions,
process limits and secret handling through the selected deployment's
configuration.

## 4. Create certificates and administrative identities

Use your organization's PKI for production. For a first test, this complete
example creates a local CA, HTTPS server certificate, AS signing certificate,
an Ed25519 KBS administrative signing key, and a separate offline bundle
approval key.
The AS key is P-256 because the CVM profile validates ES256 tokens.

```sh
umask 077
export TRUSTEE_PKI="$HOME/workspace/cvm-trustee-pki"
export CVM_BUILDER_SOURCE="$PWD"
export TRUSTEE_DNS=trustee.example.org
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
openssl genpkey -algorithm Ed25519 -out acceptance-signing.key
openssl pkey -in acceptance-signing.key -pubout -out acceptance-signing.pub
# For a standalone test deployment, install only the runtime files.
sudo install -m 0644 ca.pem as-ca.pem as-chain.pem as-public.pem \
  kbs-admin.pub acceptance-signing.pub \
  /etc/cvm-trustee/pki/
sudo install -m 0640 -o root -g cvm-trustee server.pem server.key as.key \
  /etc/cvm-trustee/pki/
cd "$CVM_BUILDER_SOURCE"
```

The standalone KBS configuration reads `server.pem`, `server.key`, `as.key`,
`as-chain.pem`, `as-ca.pem`, and `kbs-admin.pub` from `/etc/cvm-trustee/pki`.
Keep the CA private keys and `kbs-admin.key` outside that directory. The `sudo`
commands are only a native-host example; use Kubernetes Secrets or the selected
deployment's secret manager for CoCo.

The AS signing root must issue only AS signing identities. Never add the transport
CA to KBS `trusted_certs_paths`: a builder client certificate must not be able to
sign an accepted attestation token. Test direct resource retrieval with EARs
signed by builder, admin, server, and an unrelated signer; every request must fail.

Retain `ca.key`, `as-ca.key`, and `kbs-admin.key` with the deployment operator,
and keep `acceptance-signing.key` with the offline acceptance authority.
Give builders only the HTTPS CA and a pre-issued resource-administration bearer
token (§6). Public `as-public.pem` and `ca.pem` go into CVM profile inputs.
No backend private key or administrative token belongs in a CVM, application
vault, or OCI delivery.

### Keep acceptance and production on the same Trustee identity

The generic bundle measures `kbs_url`, the KBS transport CA, and the exact AS
token-signing public key. An acceptance KBS with disposable PKI cannot be handed
an already accepted bundle and then replaced by production: the guest will
reject the production TLS or appraisal signature (and a different URL changes
the measured contract).

The preferred promotion model is to test the eventual production Trustee while
it is isolated from production clients, then open that same endpoint and state
after acceptance. Keep the final DNS name in the profile from the first build.
If policy requires a separate acceptance instance, use isolated DNS/routing so
the measured production hostname reaches that instance, issue its server
certificate from the same pinned transport CA, and provision the same AS signing
key. Promote the tested KBS configuration, AS policies, reference records,
resource policy, and administrative state as one controlled release. Sharing
the AS private key expands its exposure, so prefer testing the actual production
instance over cloning it.

`prepare_lab.py` intentionally creates disposable PKI. It can qualify protocol
behavior for a profile built with that lab identity, but its approval is not
portable to a bundle built with production `kbs_cert`, `as_public_key`, or URL.

For production acceptance, stage the eventual Trustee identity in a protected
directory and run the same local harness against it before the endpoint is
opened. The directory must contain `ca.pem`, `as-ca.pem`, `as-public.pem`,
`server.{key,pem}`, `as.{key,pem}`, `kbs-admin.{key,pub}`, plus the
`builder`, `admin`, and `untrusted` test identity key/certificate pairs used by
the negative signer tests. Private keys must be owner-only (for example, mode
`0600`). Prepare the lab with the measured production hostname:

```sh
PYTHONPATH=nvflare/lighter/cc/image_builder python3 \
  tests/integration_test/lighter/cc/image_builder/prepare_lab.py \
  /srv/cvm/acceptance --platform intel_tdx \
  --identity-directory /secure/staged-trustee-identity \
  --kbs-url https://trustee.example.org:8443
```

Populate the public guest inputs and reviewed references listed in the CVM
Builder README, build Trustee at the pinned revision in
`/srv/cvm/acceptance/trustee-source`, and route
`trustee.example.org` to this isolated host from the candidate guest. Start the
local staged service from the repository root:

```sh
sudo env PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder" \
  python3 tests/integration_test/lighter/cc/image_builder/lab_kbs.py \
  /srv/cvm/acceptance
```

The acceptance runner deliberately sets `CVM_BACKEND_LOCAL=1`: NRAS fault
injection and SNP KDS blocking modify only this local staged service and must
not run against an open production instance. After all checks pass, promote the
same transport CA, AS signer, measured hostname, KBS configuration, policies,
references, and administration state. If any of those measured identities
change, create a new profile version and repeat build, acceptance, approval, and
vault sealing.

## 5. Configure upstream KBS

For a standalone deployment, copy [trustee/kbs.json](trustee/kbs.json) as the
base and adjust its listener, certificate paths, selectors, bundle-scoped ACL,
and verifier settings. For CoCo, merge the same settings into its existing configuration.
This is the upstream v0.22 schema. Its `cvm-policy` role can publish resource policy and query named
references. It cannot replace AS policies, register references, or mutate native
KBS resources. The separate `cvm-resources` role permits native resource POST and DELETE under
canonical `keys/<build_id>/<binding_id>` paths. Because native POST replaces and
DELETE removes any key the role can reach, prefer one bundle-scoped role per
approved bundle: `./cvmctl admin acl <build_id>` prints the `cvm-resources-<build_id>`
ACL entry to merge into `kbs.json`, and the builder accepts that role name for
signed tokens as well. It cannot publish policy. This upstream ACL is
endpoint-based: do not describe its resource token as create-only.

The checked-in file contains only a placeholder bundle-scoped ACL. Replace
`EXAMPLE_BUILD_ID` with the exact approved build ID; it intentionally omits the
broad `cvm-resources` role. For production SNP, add the upstream
`snp_verifier` `OfflineStore` that points at the authenticated per-host VCEK
store described in the build guide. For an NVIDIA CC profile, add the reviewed
`nvidia_verifier` remote endpoint. CPU-only TDX needs neither entry. Treat these
as required mode-specific edits, then validate the complete deployed JSON with
`cvmctl preflight trustee` before generating the deployment receipt.

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

CoCo guests omit an attestation-policy selector and continue to use Trustee's
`default` policies. Every CVM profile sets a non-default
`attestation_policy_selector` and `attestation_policy_id`. Configure the shared
KBS to map each selector to its policy ID:

```json
"attestation_service": {
  "type": "coco_as_builtin",
  "policy_id_map": {
    "cvm-cpu-2026-09-r5": ["cvm-cpu-2026-09-r5"],
    "cvm-gpu-2026-09-r5": ["cvm-gpu-2026-09-r5"]
  }
}
```

The checked-in CoCo service uses TOML and `coco_as_grpc`; the same native map is:

```toml
[attestation_service.policy_id_map]
cvm-cpu-2026-09-r5 = ["cvm-cpu-2026-09-r5"]
cvm-gpu-2026-09-r5 = ["cvm-gpu-2026-09-r5"]
```

Install the reviewed AS policy files, update this map, and restart KBS in one
coordinated maintenance window before approving either profile. Map each CVM
selector to exactly one policy ID because CoCo AS evaluates one policy set per
evidence class. Existing CoCo guests send no selector, so this table does not
change their default appraisal path.

An unknown selector is rejected. Omitting the selector selects the CoCo default,
so a CVM profile is forbidden from naming `default`. Trustee loads CPU and GPU
policies for a mapped ID from `<policy-id>_cpu.rego` and
`<policy-id>_gpu.rego`, and emits the selected policy ID in
`ear.appraisal-policy-id`. The deployment receipt hashes each installed file and
records the exact `policy_id_map` entry.

Each profile's AS policy reads one profile-scoped RVPS object containing its
values and per-value expiry deadlines. CPU and GPU profiles therefore share the
same RVPS and KBS endpoint without sharing an allowlist. Leave the existing
CoCo default policies and reference records unchanged.

Before starting KBS, retain CoCo's default files and install each rendered CVM
bundle policy under its mapped ID:

```sh
sudo install -m 0600 -o cvm-trustee -g cvm-trustee \
  /srv/cvm-cpu/attestation_policy.rego \
  /var/lib/cvm-trustee/storage/attestation_service_policy/cvm-cpu-2026-09-r5_cpu.rego
sudo install -m 0600 -o cvm-trustee -g cvm-trustee \
  /srv/cvm-gpu/attestation_policy.rego \
  /var/lib/cvm-trustee/storage/attestation_service_policy/cvm-gpu-2026-09-r5_cpu.rego
sudo install -m 0600 -o cvm-trustee -g cvm-trustee \
  /srv/cvm-gpu/gpu_attestation_policy.rego \
  /var/lib/cvm-trustee/storage/attestation_service_policy/cvm-gpu-2026-09-r5_gpu.rego
```

On a new deployment, also install the initial deny-all resource policy:

```sh
sudo install -m 0600 -o cvm-trustee -g cvm-trustee trustee/resource_policy.rego \
  /var/lib/cvm-trustee/storage/kbs/resource-policy.rego
```

Do not reset an existing deployment's published policy. Require this file to
exist at startup, before accepting traffic. For CoCo, apply this check in its
Kubernetes init container; for a native service, make it a start prerequisite.
Upstream's built-in fallback resource policy is not the CVM authorization policy
and must not be used when a policy volume is missing.

For a GPU profile install that bundle's generated GPU policy beside its CPU
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
Production SNP acceptance requires upstream's offline certificate store. Add
this alongside the existing DCAP/NVIDIA verifier settings:

```json
"snp_verifier": {
  "vcek_sources": [
    {"type": "OfflineStore"},
    {"type": "KDS"}
  ]
}
```

Mount the populated store at
`/opt/confidential-containers/attestation-service/kds-store/vcek`. The first
source lets the KDS-loss acceptance case continue with the exact host/TCB VCEK;
the optional second source retrieves collateral for a missing entry. Omit KDS
only for an air-gapped deployment. [BUILD_GUIDE.md](BUILD_GUIDE.md#1-prepare-the-build-host)
shows the required layout and preflight check.

The GPU Rego policy requires successful NRAS report/RIM checks, secure boot,
disabled debug, and approved driver/VBIOS versions. It uses only upstream claims;
it does not rely on custom `verifier` or `arch` marker fields. Optional version,
device-type and schema fields must agree when present. The KBS resource policy
requires fresh CPU/GPU appraisals, exact device count and distinct GPU identities.

## 6. Use native Trustee resource administration

The trusted token issuer creates a signed JWT accepted by the existing KBS admin
identity provider. For the example ACL, use
`role: cvm-resources-<exact-build-id>`,
`iss: cvm-builder`, `aud: coco-trustee`, and valid `iat`, `nbf`, and `exp` claims.
Set its validity to cover the build, at most 30 days, and rotate it through your
existing credential workflow; the builder refuses expired tokens and tokens whose
`exp` lies more than 30 days after `iat` before contacting Trustee. Keep the
issuer's signing key off build workers; possession of that key would allow
minting tokens for other roles.

The standalone operator can mint the token with the checked-in client. Use the
bundle-scoped role whose ACL is printed by `cvmctl admin acl BUILD_ID`:

```sh
build_id=cvm-EXACT_BUILD_ID
./cvmctl admin acl "$build_id"
sudo ./cvmctl admin token \
  --private-key /protected/kbs-admin.key \
  --role "cvm-resources-$build_id" \
  --lifetime 3600 \
  --output ./credentials/kbs-resource-token.jwt
```

The command permits at most 30 days, writes mode `0600`, and never contacts KBS.
Transfer only the token to the build worker. Do not transfer `kbs-admin.key`.

Configure the build worker's `cvm_project.yml`:

```yaml
trustee:
  url: https://trustee.example.org:8443
  ca: ./credentials/kbs-ca.pem
  admin_token_file: ./credentials/kbs-resource-token.jwt
  expiry_warning_seconds: 604800
approval:
  public_keys:
    - ./credentials/acceptance-signing.pub
```

The token file contains only the signed JWT, with restricted file permissions.
Paths resolve relative to this YAML. `cvmctl vault` checks the token syntax and lifetime
locally before retrieving the generic image or sealing a vault, then reads and
checks it again when uploading the 64-byte key with
`POST /kbs/v0/resource/keys/<build_id>/<binding_id>`. Operators may therefore
rotate a still-valid token during a long build. Set `expiry_warning_seconds` to
the maintenance lead time (seven days by default) and rotate the token before a
denial. An expiring RVPS approval requires a new profile version, fresh
acceptance, and approval; importing the same profile cannot extend it. Guest retrieval still requires
successful attestation and resource policy authorization; administration
credentials are never delivered to guests.

Native POST permits replacement. Builds always seal fresh vaults and issue one
upload attempt; they do not automatically retry an uncertain request. Preserve
`provisioning.json` and `build_failure.json` for operator recovery. Native DELETE
has no permanent tombstone, so operators must fence uploads before revocation and
preserve deletions when restoring backups. These are upstream Trustee lifecycle
responsibilities in either deployment path; no CVM key adapter or reconciliation
daemon is required.

### Diagnose resource administration failures

The former `key_service.py`, `ResourceStore` and `approved-bundles.json` are no
longer in the request path. A native resource upload goes directly to upstream KBS.
The builder intentionally omits response bodies, tokens and keys from errors;
use its failure timestamp and the existing KBS deployment's restricted logs to
identify the server-side cause.

| Failure | Check |
|---|---|
| Authentication or authorization denied | Token signature, issuer/audience, expiry, `cvm-resources` role and the endpoint ACL. A policy-role token must not upload or delete resources. |
| Backend filesystem/permission error | KBS's actual container UID/GID, volume ownership, parent-directory traversal permissions and writable mounts for `storage/repository` and `storage/kbs`. Keep AS policies and approved references operator-controlled. |
| TLS, connection or backend availability error | Configured CA/hostname, endpoint reachability and the running KBS container's status/logs. Do not disable TLS verification. |

Inspect the existing CoCo pod's KBS container logs (or the operator-managed
native process logs), rather than looking for a CVM key-service journal. Keep
those logs private: upstream diagnostics may contain resource paths or request
details. Correct ownership/mounts through the deployment configuration; do not
make storage world-writable or broaden the token's role. After an uncertain
upload, reconcile that resource through the operator workflow before retrying;
a new POST can replace it.

## 7. Install references and policies

Before the first build, follow the workflow for the enabled platform:
[TDX discovery and approval](BUILD_GUIDE.md#discover-and-approve-tdx-tcb-references)
or [SNP reference inputs and finalization](BUILD_GUIDE.md#snp-reference-inputs-and-finalization).
TDX uses a temporary TD and the TDX-only `inspect-tcb` command. SNP requires
administrator-approved TCB/configuration inputs before construction; deferred
finalization adds the real CVM's launch measurement without approving those
inputs. Both workflows separate initial TCB approval from the exact bundle's
later boot measurements and acceptance. Unknown reference keys, including JSON
`_comment_*` fields, are rejected before construction; keep approval notes
separately.

Stop the relevant KBS instance while changing its approved AS policies or RVPS
references. Import a finalized bundle's reviewed values with an operator-chosen
expiry:

```sh
sudo -u cvm-trustee /opt/cvm-builder/cvmctl references /path/to/bundle \
  --store /var/lib/cvm-trustee/storage/reference_value \
  --state /var/lib/cvm-trustee/admin \
  --expires 2026-12-01T00:00:00Z
```

Install the reviewed builder and its Python environment under
`/opt/cvm-builder` (or another path traversable by `cvm-trustee`) before using
this command. Do not point the service user at a root-only source checkout or
virtual environment.

The import writes one upstream RVPS record named `cvm_profile_<digest>` per
security profile. Its value contains the profile's references and per-value
deadlines. The rendered AS policy queries only that record and enforces every
deadline because v0.22 RVPS does not itself reject expired metadata. Existing
approvals cannot be silently broadened or renewed. Missing or expired approvals
deny appraisal. Re-running the import for the same profile preserves both its
per-value deadlines and existing record-level expiration, even when a later
`--expires` is supplied. To extend a deadline, create a new profile version,
repeat acceptance, and approve its new record. Measurements remain in Trustee
property storage.

Apply configuration through the selected deployment's rollout mechanism. Run
`./cvmctl preflight trustee` on the trusted deployment host and confirm the
initial deny-all resource policy exists before accepting traffic. The repository
contains no Trustee systemd services to install or enable.

Use a fresh profile version and rebuild/reapprove bundles when migrating from
the earlier backend. Its files, policies, receipts, and guest binaries are not
wire- or storage-compatible with v0.22. Existing hardware acceptance does not
approve the new profile.

## 8. Verify the deployment and enable a bundle

Administration runs beside the same local_fs storage. Configure `admin.json`:

```json
{
  "url": "https://trustee.example.org:8443",
  "ca": "/etc/cvm-trustee/pki/ca.pem",
  "admin_private_key": "/protected/kbs-admin.key",
  "admin_role": "cvm-policy",
  "admin_issuer": "cvm-builder",
  "admin_audience": "coco-trustee",
  "resource_policy_file": "/var/lib/cvm-trustee/storage/kbs/resource-policy.rego",
  "policy_lock": "/var/lib/cvm-trustee/admin/resource-policy.lock",
  "state": "/var/lib/cvm-trustee/admin",
  "trustee_binary": "/opt/cvm-trustee/bin/kbs",
  "trustee_build": "/etc/cvm-trustee/trustee-build.json",
  "deployment_receipt": "/etc/cvm-trustee/deployment_receipt.json",
  "approval_public_keys": ["/etc/cvm-trustee/pki/acceptance-signing.pub"]
}
```

`approval_public_keys` lists the acceptance authorities whose Ed25519 signature
on a bundle's `approval.json` counts as production approval; `admin install`
refuses unsigned or foreign-signed receipts. The paths above are for the
standalone native deployment in §2. For an existing compatible CoCo deployment,
the Trustee administrator supplies the configured binary, immutable image
identity, and build-provenance files. Retain the image identity and source
provenance beside the binary extracted from that image, and add its immutable
`trustee_image_id` to `admin.json`. The image deployment receipt contains:

```json
{
  "trustee_commit": "512fed65642015b849f38fb13bfdec7806639987",
  "source_clean": false,
  "build_profile": "nvflare-coco-service-v1",
  "source_patch_sha256": "1c62128cc044499a4c70172f2820f0083ccf205fed06562f7caed616b3230414",
  "actix_http_version": "3.13.3",
  "actix_http_crate_sha256": "11004b0e9b44b4eb3d15e0c3132b96fb178c7e50a74758b2f17bb9cc9a7fb4f6",
  "actix_http_source_sha256": "ea345db9b3fe2346f74f99cdf0d6ae0e07b84f906464af766e9b5931002ad767",
  "kbs_image_id": "sha256:KBS_IMAGE_ID",
  "policy_selection_tested": true,
  "unauthorized_administration_denied": true,
  "policy_id_map": {
    "cvm-cpu-2026-09-r5": ["cvm-cpu-2026-09-r5"],
    "cvm-gpu-2026-09-r5": ["cvm-gpu-2026-09-r5"]
  },
  "immutable_as_policies": {
    "cvm-cpu-2026-09-r5_cpu": "SHA256_OF_CPU_POLICY",
    "cvm-gpu-2026-09-r5_cpu": "SHA256_OF_GPU_PROFILE_CPU_POLICY",
    "cvm-gpu-2026-09-r5_gpu": "SHA256_OF_GPU_POLICY"
  }
}
```

A clean native server instead records `"source_clean": true` in both build and
deployment receipts and omits the CoCo recipe and image fields. Mixing the clean
binary record with the patched CoCo image, or naming a different running image,
fails before policy publication.

After testing the actual native deployment, generate that clean receipt from
the exact KBS configuration and installed policy bytes:

```sh
sudo ./cvmctl admin deployment-receipt \
  /etc/cvm-trustee/trustee-build.json /etc/cvm-trustee/kbs.json \
  --policy cvm-cpu-2026-09-r5_cpu=/var/lib/cvm-trustee/storage/attestation_service_policy/cvm-cpu-2026-09-r5_cpu.rego \
  --policy-selection-tested \
  --unauthorized-administration-denied \
  --output /etc/cvm-trustee/deployment_receipt.json
```

Repeat `--policy NAME=FILE` for every installed CVM CPU/GPU AS policy. The two
test flags are explicit operator attestations, not tests performed by the
generator. Set them only after the live HTTPS suite proves mapped selector
selection and denial of policy-role resource mutation. The command copies the
pinned deployment provenance, hashes the supplied policy files, and records the
exact `policy_id_map`; it rejects an incomplete patched-server provenance
record, and `admin install` verifies those fields again. Initial acceptance
tests run against the policies and map staged manually in §§5 and 7; the
resulting receipt then authorizes `admin install --candidate` for the exact
tested deployment.

`policy_lock` must be shared by every publisher for this Trustee instance and
remain outside directories scanned by Trustee for Rego policies. On CoCo it is
the same file used by the service scripts. CVM
administration reads the active policy while holding this lock, replaces only
the marked CVM fragment, and publishes the merged bytes. Existing CoCo release
rules and the default policy remain intact.

Set acceptance fields only after testing the actual deployment. Verify denied
AS-policy replacement and native resource writes even with the policy publisher's
credentials; builder/admin/transport certificates cannot sign accepted EARs;
wrong, stale, unmapped-selector, cross-profile, cross-vault or incomplete GPU
appraisals deny release; key retry,
revoke and restore behavior remains correct. Verify the measured guest and KBS
emit and accept the expected mapped CPU/GPU policy while an unmodified CoCo
guest still receives the default policy. Complete the bundle's
hardware acceptance before production approval.

For the pinned v0.22.0 configuration, the live HTTPS role-boundary test uses a
valid preissued `cvm-policy` token and observes **401** for native resource
POST, PUT and DELETE. It first confirms that the same token can publish the
unchanged resource policy, then checks each denied request leaves resource and
policy bytes unchanged. This exercises the server ACL; the signing client's
local role guard alone does not establish backend denial. Do not reuse the
removed backend's POST-403/PUT-DELETE-405 expectations. When qualifying another
backend revision or proxy, record its actual denial codes and verify the same
absence of mutation; an arbitrary error response is not proof of authorization.
The executable check is
[`test_policy_role_cannot_mutate_resources_or_replace_as`](../../../../tests/integration_test/lighter/cc/image_builder/test_http.py).

```sh
sudo ./cvmctl admin install admin.json /path/to/approved/bundle
```

`GET /kbs/v0/resource-policy` now returns policy IDs, not encoded policy bytes.
Administration checks that listing and verifies the exact bytes in the shared
`kbs/resource-policy.rego` file. Named references are queried at
`GET /kbs/v0/reference-value/<name>`. Do not reuse old response decoders.

Configure the vault builder's scoped resource token in `cvm_project.yml` and
run `cvmctl vault` normally. Only key creation occurs per vault; measurements and
resource-policy rules are registered per generic bundle.

## 9. Revoke, retire and restore

First stop or fence builds targeting the resource. With a resource-role config
containing `url`, `ca` and `admin_token_file`, revoke a vault through native KBS:

```sh
python3 -m cvm admin revoke resources.json keys/BUILD_ID/BINDING_ID
```

The token must carry the `cvm-resources` role. If a trusted operator instead uses
an `admin_private_key` configuration to sign the token, set
`"admin_role": "cvm-resources"` explicitly. The default `cvm-policy` role is for
policy publication; resource requests with that signing configuration are rejected
locally before reading the key or contacting Trustee. Preissued tokens remain
subject to Trustee's signature and ACL checks.

This sends `DELETE /kbs/v0/resource/keys/<build_id>/<binding_id>`. It prevents
future retrieval while the resource is absent; an authorized POST can recreate
it. Token revocation/expiry and backup recovery belong to the Trustee operator.

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
apply outstanding legacy revocations through upstream Trustee and preserve them in
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
