# Set up Trustee for CVM vault keys

This guide deploys the key backend used by [CVM Build and Vault Build](BUILD_GUIDE.md).
Trustee authenticates a running CVM and releases its vault key only when the
hardware evidence, approved CVM measurements, and vault binding satisfy policy.
Set up this backend before building a production CVM profile.

The deployment contains three components in one KBS process: the Key Broker
Service (KBS), Attestation Service (AS), and Reference Value Provider Service
(RVPS). The builder's separate **key service** accepts key uploads from authorized
builders. Both processes use the same filesystem repository; KBS can only read it.
The current key service supports 64-byte vault keys. Support for other key types
would require extending its validation and resource namespace.

## 1. Choose the host and addresses

Use a trusted x86-64 Linux server. The backend itself does not need SNP, TDX, or a
GPU. It needs persistent storage, a synchronized clock, HTTPS connectivity to
Intel/AMD collateral services, and access from the builders and CVMs. A production
backend should have its own administrative boundary rather than run on an
untrusted CVM host.

Examples below use Ubuntu 26.04 and these locations:

| Item | Example |
| --- | --- |
| KBS address used by CVMs | `https://kbs.example.org:8443` |
| Key service address used by builders | `https://kbs.example.org:9443` |
| Installed builder modules | `/opt/cvm-builder` |
| KBS binary | `/opt/cvm-trustee/bin/kbs` |
| Certificates and configuration | `/etc/cvm-trustee` |
| Vault keys | `/var/lib/cvm-trustee-resources` |
| Permanent revocations | `/var/lib/cvm-trustee-revocations` |
| AS, RVPS, policy and publisher state | `/var/lib/cvm-trustee` |

Replace `kbs.example.org` with a DNS name that resolves from both the build host
and the CVM. A guest's `localhost` is the guest itself. For a disposable backend
on the QEMU host, `10.0.2.2` is the host address seen through QEMU user networking;
include that IP in the server certificate if using it.

Allow inbound TCP 8443 from CVM networks and TCP 9443 from builder/admin networks.
The backend needs DNS, time synchronization, and outbound HTTPS to collateral
services. Guest NTS time synchronization also needs TCP 4460 and UDP 123.

Run the following commands from this repository's `image_builder` directory in
one shell. Keep the source location for later installation:

```sh
export CVM_BUILDER_SOURCE="$PWD"
export TRUSTEE_SOURCE="$HOME/workspace/trustee-cvm"
export TRUSTEE_DNS=kbs.example.org
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates openssl \
  pkg-config libssl-dev libclang-dev clang cmake protobuf-compiler \
  libprotobuf-dev libtss2-dev gnupg chrony python3-pip
```

Disable swap and piped core collectors on this dedicated backend. The supplied
services check these settings before handling keys. Also remove/disable the
host's swap entries or swap units so the setting persists after reboot.

```sh
sudo swapoff -a
printf 'kernel.core_pattern=/dev/null\nfs.suid_dumpable=0\n' |
  sudo tee /etc/sysctl.d/90-cvm-trustee.conf >/dev/null
sudo sysctl --system
chronyc waitsync 30 0.5 1000 1
```

## 2. Build the compatible Trustee and client

This builder uses Trustee commit
`a2570329cc33daf9ca16370a1948b5379bb17fbe` with the repository's boundary patch.
An arbitrary upstream image does not include that patch. It selects the explicit
AS policy, prevents native KBS key writes and AS-policy replacement, enforces
token freshness, caches verified SNP VCEKs, and backports NVIDIA composite
evidence onto guest-components `591d0bb45cd7a2c66f3778428940c40f7eec3b7d`.
Sample TEEs and unsupported SEV/TPM verifiers return errors instead of panicking.

Install Intel's quote-verification development/runtime packages. The example
uses the `resolute` repository for Ubuntu 26.04, as on the TDX test host:

```sh
curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key |
  sudo gpg --dearmor --yes -o /usr/share/keyrings/intel-sgx.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx.gpg] https://download.01.org/intel-sgx/sgx_repo/ubuntu resolute main' |
  sudo tee /etc/apt/sources.list.d/intel-sgx.list >/dev/null
sudo apt-get update
sudo apt-get install -y libsgx-dcap-quote-verify-dev \
  libsgx-dcap-quote-verify libsgx-dcap-default-qpl
```

Use the checkout's Rust toolchain, or a tested compatible newer toolchain. With
[rustup](https://rustup.rs/) installed, the checkout selects its toolchain
automatically. The pinned checkout declares Rust 1.85.1; the Ubuntu 26.04 test
host also built it with Rust 1.93.1.

```sh
git clone https://github.com/confidential-containers/trustee.git "$TRUSTEE_SOURCE"
git -C "$TRUSTEE_SOURCE" checkout a2570329cc33daf9ca16370a1948b5379bb17fbe
python3 scripts/patch_trustee.py "$TRUSTEE_SOURCE"
cd "$TRUSTEE_SOURCE"
cargo build --locked --release -p kbs --bin kbs \
  --no-default-features --features coco-as-builtin
cargo build --locked --release -p kbs-client --bin kbs-client \
  --no-default-features \
  --features tdx-attester,snp-attester,nvidia-attester,kbs_protocol/background_check,kbs_protocol/passport,kbs_protocol/rust-crypto
cd "$CVM_BUILDER_SOURCE"
```

Apply the patch once to a clean checkout. It prints a digest and writes
`cvm-boundary.patch`, including the companion `cvm_guest.patch`. The script
clones the exact guest-components pin into `cvm_guest/` and patches the evidence
collector. Retain both source trees and patches. The client uses the measured
NVAT CLI only to collect raw evidence; it does not call NRAS from the guest.
Retain the boundary digest for the CVM profile and deployment receipt.

```sh
sha256sum "$TRUSTEE_SOURCE/cvm-boundary.patch"
python3 scripts/trustee_provenance.py "$TRUSTEE_SOURCE" \
  "$TRUSTEE_SOURCE/target/release/kbs" trustee_build.json
sudo install -d /opt/cvm-trustee/bin /opt/cvm-builder
sudo install -m 0755 "$TRUSTEE_SOURCE/target/release/kbs" /opt/cvm-trustee/bin/kbs
sudo cp -a builder scripts /opt/cvm-builder/
sudo python3 -m pip install --target /opt/cvm-builder -r requirements.txt
# Alternative to the pip command:
# sudo uv pip install --target /opt/cvm-builder -r requirements.txt
mkdir -p inputs
install -m 0755 "$TRUSTEE_SOURCE/target/release/kbs-client" inputs/kbs-client
```

For TDX, configure `/etc/sgx_default_qcnl.conf` for your approved PCCS/collateral
service and make it readable by the service account. The KBS service supplies a
writable collateral cache. For SNP, allow HTTPS to `kdsintf.amd.com`; the first
appraisal fetches and verifies the chip/TCB-specific VCEK. Subsequent appraisals
can use its in-memory cache until KBS restarts. These paths still verify
signatures, validity periods, and TCB status.

## 3. Create the service account and storage

```sh
id cvm-trustee >/dev/null 2>&1 || sudo useradd --system --home /nonexistent \
  --shell /usr/sbin/nologin cvm-trustee
sudo install -d -m 0750 -o root -g cvm-trustee /etc/cvm-trustee \
  /etc/cvm-trustee/pki /etc/cvm-trustee/as-policies /etc/cvm-trustee/as-policies/opa
sudo install -d -m 0700 -o cvm-trustee -g cvm-trustee \
  /var/lib/cvm-trustee-resources /var/lib/cvm-trustee-resources/default \
  /var/lib/cvm-trustee-revocations \
  /var/lib/cvm-trustee /var/lib/cvm-trustee/as /var/lib/cvm-trustee/policy \
  /var/lib/cvm-trustee/rvps /var/lib/cvm-trustee/admin
sudo -u cvm-trustee sh -c 'umask 077; test -e /var/lib/cvm-trustee-revocations/approved-bundles.json || printf "%s\n" "{\"build_ids\":[]}" > /var/lib/cvm-trustee-revocations/approved-bundles.json'
```

Vault key files are plaintext inside this trusted repository. Protect backend
storage and backups with your site's encryption/access controls. The service
templates make the repository read-only to KBS and writable to the key service.
Keep revocation state outside backups that can restore an older key repository.

Install the build record:

```sh
sudo install -m 0644 -o root -g root trustee_build.json /etc/cvm-trustee/trustee_build.json
```

 `admin_install` compares its source revision/patch digest
with the CVM contract and checks the installed binary's digest. Retain the trusted
build record with release evidence; copying strings into an acceptance receipt
is insufficient.

## 4. Create certificates and administrative identities

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

## 5. Configure KBS, key service, and immutable policies

Create `/etc/cvm-trustee/kbs.json` with the following content:

```json
{
  "http_server": {
    "sockets": ["0.0.0.0:8443"],
    "insecure_http": false,
    "private_key": "/etc/cvm-trustee/pki/server.key",
    "certificate": "/etc/cvm-trustee/pki/server.pem"
  },
  "attestation_token": {
    "insecure_key": false,
    "trusted_certs_paths": ["/etc/cvm-trustee/pki/as-ca.pem"]
  },
  "admin": {
    "insecure_api": false,
    "auth_public_key": "/etc/cvm-trustee/pki/kbs-admin.pub"
  },
  "policy_engine": {"policy_path": "/var/lib/cvm-trustee/policy/resource-policy.rego"},
  "attestation_service": {
    "type": "coco_as_builtin",
    "work_dir": "/var/lib/cvm-trustee/as",
    "policy_engine": "opa",
    "attestation_token_broker": {
      "type": "Ear",
      "duration_min": 5,
      "policy_dir": "/etc/cvm-trustee/as-policies",
      "signer": {
        "key_path": "/etc/cvm-trustee/pki/as.key",
        "cert_path": "/etc/cvm-trustee/pki/as-chain.pem"
      }
    },
    "rvps_config": {
      "type": "BuiltIn",
      "storage": {"type": "LocalJson", "file_path": "/var/lib/cvm-trustee/rvps/references.json"}
    }
  },
  "plugins": [{"name": "resource", "type": "LocalFs", "dir_path": "/var/lib/cvm-trustee-resources"}]
}
```

Generate `key-service.json` from the client certificates. Fingerprints are SHA-256
of DER certificate bytes, written as lowercase hex without colons:

```sh
export TRUSTEE_BUILDER_FP=$(openssl x509 -in "$TRUSTEE_PKI/builder.pem" -outform DER | sha256sum | cut -d ' ' -f 1)
export TRUSTEE_ADMIN_FP=$(openssl x509 -in "$TRUSTEE_PKI/admin.pem" -outform DER | sha256sum | cut -d ' ' -f 1)
python3 - <<'PY' | sudo tee /etc/cvm-trustee/key-service.json >/dev/null
import json, os
print(json.dumps({
    "listen": "0.0.0.0", "port": 9443,
    "resources": "/var/lib/cvm-trustee-resources",
    "state": "/var/lib/cvm-trustee-revocations",
    "client_ca": "/etc/cvm-trustee/pki/ca.pem",
    "cert": "/etc/cvm-trustee/pki/server.pem",
    "key": "/etc/cvm-trustee/pki/server.key",
    "certificate_roles": {
        os.environ["TRUSTEE_BUILDER_FP"]: "builder",
        os.environ["TRUSTEE_ADMIN_FP"]: "admin"
    }
}, indent=2))
PY
```

Install the CPU appraisal policy and a deny-by-default fallback. Use the same
policy ID in the CVM profile and KBS environment:

```sh
sudo install -m 0640 -o root -g cvm-trustee config/attestation_policy.rego \
  /etc/cvm-trustee/as-policies/opa/cvm-cpu-r2_cpu.rego
printf '%s\n' 'package policy' 'import rego.v1' \
  'default executables := 33' 'default hardware := 97' 'default configuration := 36' |
  sudo tee /etc/cvm-trustee/as-policies/opa/default_cpu.rego >/dev/null
printf '%s\n' 'CVM_AS_POLICY_ID=cvm-cpu-r2' |
  sudo tee /etc/cvm-trustee/environment >/dev/null
sudo -u cvm-trustee sh -c 'umask 077; test -e /var/lib/cvm-trustee/policy/resource-policy.rego || printf "%s\n" "package policy" "default allow = false" > /var/lib/cvm-trustee/policy/resource-policy.rego'
sudo -u cvm-trustee sh -c 'umask 077; test -e /var/lib/cvm-trustee/rvps/references.json || printf "[]\n" > /var/lib/cvm-trustee/rvps/references.json'
sudo chown root:cvm-trustee /etc/cvm-trustee/*.json /etc/cvm-trustee/environment \
  /etc/cvm-trustee/as-policies/opa/*.rego
sudo chmod 0640 /etc/cvm-trustee/*.json /etc/cvm-trustee/environment \
  /etc/cvm-trustee/as-policies/opa/*.rego
```

The pinned AS engine adds `opa/` to `policy_dir`. The `default_cpu.rego` file is
required at initialization even though the patched selector uses the explicit
policy ID. KBS selects one AS policy ID per instance; a policy-ID migration needs
a coordinated profile/backend rollout or a separate instance.

### GPU profiles: configure the backend verifier and second AS policy

Use a separate instance and a new profile version/policy ID, for example
`gpu-2026.09-r2` / `cvm-gpu-r2`. This patch explicitly supports **NRAS remote
verification on the KBS host**. Local RIM/OCSP appraisal is not enabled and there
is no fallback to guest-side appraisal. NRAS performs RIM/certificate/OCSP
checks; KBS verifies its signed overall and per-device JWTs, digest linkage,
issuer, timestamps and the exact RCAR-derived nonce.

Obtain a reviewed NVIDIA JWKS snapshot from
`https://nras.attestation.nvidia.com/.well-known/jwks.json`, verify its origin
through the site's trusted channel, and install it read-only. Pin its SHA-256 in
`/etc/cvm-trustee/nvidia.json`:

```json
{
  "mode": "remote",
  "url": "https://nras.attestation.nvidia.com/v4/attest/gpu",
  "issuer": "https://nras.attestation.nvidia.com",
  "jwks": "/etc/cvm-trustee/nras_jwks.json",
  "jwks_sha256": "REPLACE_WITH_REVIEWED_JWKS_SHA256"
}
```

Add `CVM_NVIDIA_CONFIG=/etc/cvm-trustee/nvidia.json` to the KBS environment.
Set `CVM_AS_POLICY_ID=cvm-gpu-r2` in that same file. The service account must
read both JSON files; keep them root-owned and non-writable by KBS. Allow outbound
HTTPS to NRAS from the **backend**. Unknown rotated signing keys fail closed;
review and install a new JWKS pin and restart KBS rather than accepting keys
from an attestation response.

Stage 1 renders `gpu_attestation_policy.rego` from the profile's strict
`gpu_policy.json`. Install it alongside the CPU policy from that exact bundle:

```sh
export CVM_BUNDLE=/srv/cvm/bundles/gpu-2026.09-r2/amd_sev_snp
sudo install -m 0640 -o root -g cvm-trustee "$CVM_BUNDLE/attestation_policy.rego" \
  /etc/cvm-trustee/as-policies/opa/cvm-gpu-r2_cpu.rego
sudo install -m 0640 -o root -g cvm-trustee "$CVM_BUNDLE/gpu_attestation_policy.rego" \
  /etc/cvm-trustee/as-policies/opa/cvm-gpu-r2_gpu.rego
sudo sha256sum /etc/cvm-trustee/as-policies/opa/cvm-gpu-r2_*.rego
```

`CVM_BUNDLE` here is the finalized GPU bundle directory. Record **both** digests
under `immutable_as_policies` keys `cvm-gpu-r2_cpu` and `cvm-gpu-r2_gpu` in the
deployment receipt. `admin_install` requires both. The pinned EAR broker initializes
only `default_cpu.rego`; no `default_gpu.rego` is required. A missing selected GPU
policy fails closed.

GPU reference inputs additionally require reviewed, nonempty
`gpu_driver_versions` and `gpu_vbios_versions` string lists. Import them with the
bundle through §7; never infer approval from a device's claimed version. AS
compares signed NRAS version claims with those RVPS values and requires secure
boot, debug disabled, successful measurements, nonce/report/RIM/certificate checks
and an approved supported architecture. GPU submods must be exactly `gpu0` through
`gpuN-1`, with distinct NVIDIA device identities, the same policy ID as `cpu0`,
and the affirming integer trust vector. Sample-device claims cannot satisfy this
rule. CPU-only resource rule bytes are unchanged.

## 6. Start the services

```sh
sudo install -m 0644 trustee/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cvm_key_service.service cvm_trustee_kbs.service
sudo systemctl status cvm_key_service.service cvm_trustee_kbs.service --no-pager
sudo journalctl -u cvm_key_service -u cvm_trustee_kbs -n 50 --no-pager
```

KBS automatically runs `cvm_key_reconcile.service` before serving requests.
Check the actual administrative endpoint; this version does not expose a generic
`/health` endpoint:

```sh
curl --cacert "$TRUSTEE_PKI/ca.pem" -sS -o /dev/null -w '%{http_code}\n' \
  "https://$TRUSTEE_DNS:8443/kbs/v0/resource-policy"
```

An unauthenticated request must return a rejection, never HTTP 200. Inspect the
logs if the connection fails. The initial resource policy denies every key request.

## 7. Connect the CVM profile and load approved references

On the build host, copy the public CA, AS public key, and compatible `kbs-client`
to `inputs/`. Configure these fields in the CVM profile:

```yaml
trustee_commit: a2570329cc33daf9ca16370a1948b5379bb17fbe
trustee_patch_digest: 94de3a62f7abc62664be7734667bb300fefdc6aec52ae99e4662c03678bc720e
kbs_url: https://kbs.example.org:8443
kbs_cert: ../inputs/kbs-ca.pem
as_public_key: ../inputs/as-public.pem
token_algorithm: ES256
token_issuer: null
attestation_policy_id: cvm-cpu-r2
attestation_policy: attestation_policy.rego
reference_values: ../inputs/approved-tcb-references.json
bootstrap_egress: [443, 8443]
```

Confirm `trustee_patch_digest` against the patch you built. The reference-values
input is a JSON object of **administrator-approved** platform values. For SNP it
contains the `snp_*` TCB/configuration values used by
[`config/attestation_policy.rego`](config/attestation_policy.rego); for TDX it
contains `mr_seam`, `tcb_svn`, `xfam`, and `allowed_advisory_ids`. Empty or missing
approvals fail closed. Do not approve arbitrary measurements simply because a
guest reports them. Review platform firmware/TCB endorsements and site policy.

For TDX, inspect the private reference report captured by the initial build:

```sh
python3 /opt/cvm-builder/scripts/tcb_inspect.py /restricted/reference-evidence.json
```

The output contains **unapproved candidate** `mr_seam`, `tcb_svn`, and `xfam`
values. It does not import references or verify a platform endorsement. Obtain
advisory IDs from a real signed-quote appraisal and review its TCB status before
creating the administrator-approved reference file. Keep the source report in
the restricted acceptance record; do not publish it with the bundle.

Build and finalize the CVM using [Advanced CVM Build](BUILD_GUIDE.md#3-advanced-cvm-build)
for the initial backend bring-up. Finalization adds its software measurements to
the bundle's `reference_values.json`. The backend operator then imports that
file into RVPS. This happens once per approved CVM bundle, not once per vault.

Each Trustee instance serves one security profile (the exact profile version
and contract). Use separate instances, resource stores, publisher state and policy
IDs for CPU-only, GPU, or differently approved TCB profiles. `admin_install` and
the importer enforce this isolation, including after all bundles are retired.

For LocalJson, stop KBS and import references as its service identity. Choose an
explicit reviewed expiry. Existing values must match exactly; imports neither
union values nor renew existing expiry dates. A TCB change requires a new profile
and instance. Renewal is a separate administrator review, not a side effect of
adding a bundle.

```sh
export CVM_BUNDLE=/srv/cvm/bundles/cpu-2026.09-r2/intel_tdx
sudo systemctl stop cvm_trustee_kbs
sudo -u cvm-trustee python3 /opt/cvm-builder/scripts/trustee_references.py \
  "$CVM_BUNDLE" --store /var/lib/cvm-trustee/rvps/references.json \
  --state /var/lib/cvm-trustee/admin --expires 2026-12-01T00:00:00Z
sudo systemctl start cvm_trustee_kbs
```

Monitor RVPS expiration and retire obsolete profiles. Unknown input keys are
rejected before building; JSON `_comment_*` entries are not reference values.

## 8. Verify the backend and enable a bundle

Create an operator-only `admin.json` on the backend. `admin_install` accesses
local repository/state paths, so run it there; it is not a remote-only client.

```json
{
  "url": "https://kbs.example.org:8443",
  "ca": "/etc/cvm-trustee/pki/ca.pem",
  "admin_private_key": "/root/cvm-trustee-admin/kbs-admin.key",
  "resources": "/var/lib/cvm-trustee-resources",
  "key_service_state": "/var/lib/cvm-trustee-revocations",
  "state": "/var/lib/cvm-trustee/admin",
  "deployment_receipt": "/root/cvm-trustee-admin/deployment-receipt.json",
  "trustee_binary": "/opt/cvm-trustee/bin/kbs",
  "trustee_build": "/etc/cvm-trustee/trustee_build.json"
}
```

Install `admin.json` and `kbs-admin.key` in `/root/cvm-trustee-admin` with directory
mode 0700 and file mode 0600. The deployment receipt has this structure:

```json
{
  "trustee_commit": "a2570329cc33daf9ca16370a1948b5379bb17fbe",
  "trustee_patch_digest": "94de3a62f7abc62664be7734667bb300fefdc6aec52ae99e4662c03678bc720e",
  "policy_selection_tested": false,
  "unauthorized_administration_denied": false,
  "immutable_as_policies": {
    "cvm-cpu-r2_cpu": "REPLACE_WITH_INSTALLED_CPU_POLICY_SHA256"
  }
}
```

Obtain the policy hash with:

```sh
sudo sha256sum /etc/cvm-trustee/as-policies/opa/cvm-cpu-r2_cpu.rego
```

The two flags are deliberately false in the sample. Set them to true only after
verifying policy selection and administrative denial on the deployed revision.
Specifically, verify that the selected AS policy reports its explicit ID, rejects
unapproved TCB/measurements, and cannot be replaced through the AS policy API;
unauthorized policy/RVPS writes and native KBS resource writes must fail.
For the pinned router, authenticated native resource POST returns 403; unsupported
PUT and DELETE return 405. Check each method and confirm that stored key bytes
and policy bytes remain unchanged; a non-200 status alone is insufficient. Verify
the KBS process sees the key repository and AS policy directory as read-only.
The repository's isolated HTTPS tests in `tests/test_http.py` exercise policy
readback, denied mutation, key creation/conflict, cross-vault denial, expiry,
revocation and restore. Follow [VALIDATION.md](VALIDATION.md) for the disposable
lab harness; its synthetic signing fixtures belong only in that test deployment.
Never run its policy-changing fixtures against a live key backend.

After the CVM has passed site acceptance and has `approval.json`, enable it:

```sh
cd /opt/cvm-builder
sudo ./scripts/admin_install /root/cvm-trustee-admin/admin.json "$CVM_BUNDLE"
# Preserve the service account's ownership of files created by local administration.
sudo chown -R cvm-trustee:cvm-trustee /var/lib/cvm-trustee/admin \
  /var/lib/cvm-trustee-revocations
```

For a disposable acceptance build whose profile starts with `test-`, append
`--candidate`. This allows test key provisioning before production approval;
it does not approve the bundle. A receipt about the backend is distinct from the
CVM's hardware acceptance report and `approval.json`.

`admin_install` checks the receipt, verifies RVPS references, publishes the
combined resource policy, reads it back byte-for-byte, and enables key creation
for this build ID. Each permitted key path is derived from signed guest evidence:

```text
keys/<cvm_build_id>/<hardware_bound_vault_id>
```

There is no per-vault measurement registration file. Backend state tracks
approved reusable bundles, stored keys, and permanent revocations.

## 9. Upload keys through Vault Build

On the build host, configure the builder's client certificate once in the project
root's [cvm_project.yml](cvm_project.yml). Every production vault build shares it:

```yaml
key_service:
  url: https://kbs.example.org:9443
  ca: ./inputs/key-service-ca.pem
  cert: ./inputs/builder-client.pem
  key: ./inputs/builder-client.key
```

Copy `ca.pem`, `builder.pem`, and `builder.key` from the PKI into those input
paths relative to `cvm_project.yml`, protecting the client key with mode 0600.
Remove `key_service` from per-build YAML. The builder searches upward from the
build YAML for the nearest project file; pass `--project-config /path/cvm_project.yml`
when the build YAML is outside the project. Run [Vault Build](BUILD_GUIDE.md#4-vault-build):

```sh
sudo ./vault_build.sh config/vault_build.yml
```

Vault Build generates a fresh key and binding for each sealed platform copy and
uploads the key using mutual TLS. Repeating an upload with the same path and key
is idempotent; changing the bytes for an existing path is rejected. The key and
client credentials are not packaged into the OCI delivery. The recipient follows
[USER_GUIDE.md](USER_GUIDE.md); the launched CVM retrieves its own key after CPU
attestation. CPU-only applications retain CPU-only authorization. GPU profiles
require a composite CPU/GPU EAR at KBS before key release, including during
periodic reauthorization. Failed or missing GPU evidence cannot unlock their vault.

## 10. Revoke, retire, back up, and restore

To revoke one vault, use its external `vault_manifest.json` to obtain `resource`,
then send DELETE with the **admin client** certificate. This example reads public
metadata and never prints a secret:

```sh
export VAULT_MANIFEST=/srv/cvm/delivery/intel_tdx/vault_manifest.json
export VAULT_RESOURCE=$(python3 -c 'import json,os; print(json.load(open(os.environ["VAULT_MANIFEST"]))["resource"])')
curl --fail --cacert "$TRUSTEE_PKI/ca.pem" \
  --cert "$TRUSTEE_PKI/admin.pem" --key "$TRUSTEE_PKI/admin.key" \
  -X DELETE "https://$TRUSTEE_DNS:9443/v1/resources/$VAULT_RESOURCE"
```

The key service persists a revocation tombstone before deleting the key. The
builder identity cannot revoke keys or restore a revoked identity. Revocation
blocks later key requests; it cannot erase a key already released into a CVM.
The guest's periodic authorization check stops the workload when access is denied.

Retire an entire CVM build on the backend:

```sh
cd /opt/cvm-builder
sudo ./scripts/admin_retire /root/cvm-trustee-admin/admin.json cvm-BUILD_ID
sudo chown -R cvm-trustee:cvm-trustee /var/lib/cvm-trustee/admin \
  /var/lib/cvm-trustee-revocations
```

Back up resource files, active policy/publisher state, RVPS approvals, and backend
configuration/keys through the site's protected backup process. Preserve current
revocation/retirement state separately. During a restore:

1. Stop both KBS and the key service to prevent reads or uploads during restoration.
2. Restore the resource snapshot and matching administrative configuration.
3. Retain the current revocation/retirement state; never replace it with older state.
4. Restore ownership to `cvm-trustee` and mode 0600 for resource/state files.
5. Run reconciliation, then start the services. Startup also reconciles before KBS.

```sh
sudo systemctl stop cvm_trustee_kbs cvm_key_service
# Restore the selected backup while both services are stopped.
sudo systemctl start cvm_key_reconcile
sudo systemctl start cvm_key_service cvm_trustee_kbs
```

Verify revoked resources are still denied before reopening access. After renewing
a client certificate, update its fingerprint mapping and restart the key service.
After rotating the AS signing key or CA, rebuild the CVM profile with the new
public trust material and coordinate the rollout.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Service refuses startup | Swap, `core_pattern`, file ownership, Python requirements, TLS paths, `CVM_AS_POLICY_ID`, and `journalctl`. |
| Key service returns HTTP 503 | The protected journal reports the failing operation and exception class/errno. Check repository permissions, filesystem capacity and I/O health; no keys, tokens, request bodies or exception text are logged. |
| Key service returns HTTP 409 | Build ID not enabled, conflicting existing key, revoked identity, or invalid 64-byte upload/path. |
| `admin_install` rejects the receipt | Exact Trustee commit/patch hash, installed policy hash, and completed backend checks. |
| RVPS references missing | Import finalized bundle references, restart the LocalJson KBS, and check expiration. |
| Positive CPU quote still denied | AS policy ID, required trust-vector values, TCB/configuration approvals, and exact bundle/vault binding. |
| TDX appraisal fails | QGS on the CVM host, PCCS/QCNL settings on the backend, current Intel collateral, and writable KBS cache. |
| SNP first appraisal fails | AMD KDS connectivity and valid chip/TCB endorsements; the cache is empty after restart. |
| Guest stops before requesting a key | Clock synchronization and guest DNS/NTS/NTP/KBS connectivity. |
| New certificate cannot upload | New DER SHA-256 fingerprint in `certificate_roles`, correct client CA and clientAuth usage. |

Upstream references: [pinned KBS configuration](https://github.com/confidential-containers/trustee/blob/a2570329cc33daf9ca16370a1948b5379bb17fbe/kbs/docs/config.md),
[pinned RVPS](https://github.com/confidential-containers/trustee/tree/a2570329cc33daf9ca16370a1948b5379bb17fbe/rvps),
[Intel DCAP](https://github.com/intel/SGXDataCenterAttestationPrimitives), and
[AMD SEV documentation](https://www.amd.com/en/developer/sev.html).
