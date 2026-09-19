# Build a CVM bundle and application vault

The generic CVM contains Ubuntu, the container runtime, attestation code, and the
vault unlock path. Build it once for each platform and profile version, then reuse
it for every compatible application. Build a new vault for every application
release and site.

Set up the attestation and key backend first using
[TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md). It covers existing CoCo deployment configuration, test
certificates, reference-value installation, and the `trustee` credentials
used by Vault Build.

## 1. Prepare the build host

Run from `nvflare/lighter/cc/image_builder` on an Ubuntu 26.04 SNP or TDX host.
Python is assumed to be installed. Install the Python requirements with pip or uv:

```sh
python3 -m pip install -r requirements.txt
```

```sh
uv pip install -r requirements.txt
```

Install the host tools and the Ubuntu OVMF firmware packages:

```sh
sudo apt-get update
sudo apt-get install -y qemu-system-x86 qemu-utils cloud-image-utils \
  libguestfs-tools cryptsetup-bin e2fsprogs util-linux openssh-client curl \
  git cargo docker.io ovmf gpgv ubuntu-cloudimage-keyring
```

Registry publication and retrieval use ORAS; an offline `.oci.tar` build and
materialization do not require it. Installation packages and instructions are at
<https://oras.land/docs/installation/>. Production registry releases should also
use Cosign; installation instructions are at
<https://docs.sigstore.dev/cosign/system_config/installation/>.

The tested host baseline is Ubuntu 26.04 with kernel 7.0.0-31 and QEMU 10.2.1.
Use Intel's repository for a published, supported distribution; do not substitute
an arbitrary Ubuntu codename (for example, `questing`/`plucky`) in the DCAP URL.
Follow the [Intel host setup guide](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/05/host_os_setup/)
for host enablement and the collateral service selected by your site.

For TDX, complete these steps before a full build:

1. Enable `nohibernate kvm_intel.tdx=1` in the host kernel command line and reboot.
   Instead of the `kvm_intel.tdx=1` argument, you may configure
   `options kvm_intel tdx=1` in modprobe configuration; keep `nohibernate`.
   Verify `/sys/module/kvm_intel/parameters/tdx` is `Y`. TDX module initialization
   may be lazy; absence of an early boot message alone is not a failure.
2. Install `tdx-qgs` from the supported Intel DCAP repository. In `/etc/qgs.conf`,
   uncomment/set `port = 4050` for the supplied vsock profile, then start/restart
   `qgsd`. A commented port selects a Unix socket, which does not match the sample.
3. Configure `/etc/sgx_default_qcnl.conf` for your approved PCS or caching service.
   A dedicated PCCS and personal PCS key are not universally required; use the
   credentials and registration procedure for the selected service. Multi-package
   hosts may require Intel platform registration before PCK certificates can be
   retrieved. Check that registration and collateral retrieval succeed: a 404 for
   `pckcerts` needs investigation before blaming the CVM policy.
4. Supply a reviewed direct-boot TDVF as `inputs/OVMF.inteltdx.fd`. The packaged
   `OVMF.inteltdx.ms.fd` enables Secure Boot and is not the default for an unsigned
   direct-boot kernel. The tested TDVF build uses `SECURE_BOOT_ENABLE=FALSE`; its
   source, build flags and digest are recorded in
   [VALIDATION.md](VALIDATION.md#recorded-tdx-firmware-input). TDX measurements
   authenticate this firmware and kernel through the approved launch profile.
   A Secure Boot deployment must separately validate its signed kernel/shim chain;
   `platforms.intel_tdx.shim` supplies the shim to QEMU's `-shim` option.
5. Run a minimal TD quote smoke test against the intended collateral/backend path.
   A successful local TDREPORT is not proof that QGS can generate a signed quote.

```sh
sudo ./cvmctl preflight host --firmware inputs/OVMF.inteltdx.fd \
  --quote-probe /usr/local/sbin/site_tdx_quote_probe
```

`site_tdx_quote_probe` is a site-provided executable; it is not shipped here.
For a concrete reference implementation, follow Canonical's
[TD image creation](https://github.com/canonical/tdx#create-td-image),
[TD boot procedure](https://github.com/canonical/tdx#boot-td), and
[quote generation and remote verification example](https://github.com/canonical/tdx#perform-remote-attestation).
The example uses a small Ubuntu TD without a GPU. After installing its guest
attestation tools and configuring an Intel Tiber Trust Services API key, run
inside that TD:

```sh
umask 077
trustauthority-cli evidence --tdx --config ./config.json > evidence.json
trustauthority-cli token --config ./config.json > token.txt
test -s token.txt
```

The first command exercises QGS; the second requests verification and a token.
Token presence alone does not establish an acceptable appraisal: validate its
signature, nonce, expiry and TCB result using the reference workflow. This is an
optional external-service smoke test, not a replacement for CoCo Trustee or CVM
reference approval. A site's automated probe must perform those checks against
its intended verifier and collateral channel, exit zero only on success, and
stop its temporary TD. The preflight runs it with a 180-second limit and exits
nonzero if it is omitted. On failure, inspect `journalctl -u qgsd`, collateral
access and platform registration before building/sealing.

For a previously working host that stops producing quotes, or a verified quote
that is appraised as `OutOfDate`, follow
[Intel TDX troubleshooting](TDX_TROUBLESHOOTING.md). It covers QGS cache-expiry
recovery and the separate Trustee `standard`/`early` collateral setting.

For the plain construction VM and SNP, verify the packaged firmware paths:

```sh
test -r /usr/share/ovmf/OVMF.fd
test -r /usr/share/ovmf/OVMF.amdsev.fd
```

Authenticate the Ubuntu 26.04 cloud-image checksum before using the base image.
The signing keys come from `ubuntu-cloudimage-keyring`, installed through the
build host's authenticated Ubuntu package repositories. This trusted keyring is
independent of the image download and its accompanying checksum files; do not
replace it with a key supplied by the image mirror. See Ubuntu's
[cloud-image verification guidance](https://documentation.ubuntu.com/public-images/public-images-how-to/verify-image-checksum/).

Run the following block as one command. It stops on a missing or invalid
signature, a missing or duplicate image checksum, or an image digest mismatch:

```sh
(
  set -eu
  mkdir -p inputs
  cd inputs
  base_image_name=ubuntu-26.04-server-cloudimg-amd64.img
  base_image_url=https://cloud-images.ubuntu.com/releases/26.04/release
  curl -fL "$base_image_url/SHA256SUMS" -o SHA256SUMS
  curl -fL "$base_image_url/SHA256SUMS.gpg" -o SHA256SUMS.gpg
  gpgv --keyring /usr/share/keyrings/ubuntu-cloudimage-keyring.gpg \
    SHA256SUMS.gpg SHA256SUMS
  awk -v image="$base_image_name" '
    $2 == image || $2 == "*" image { print; found++ }
    END { if (found != 1) exit 1 }
  ' SHA256SUMS > "$base_image_name.sha256"
  curl -fL "$base_image_url/$base_image_name" -o "$base_image_name"
  sha256sum --check --strict "$base_image_name.sha256"
)
```

Proceed with construction only after this block succeeds. Retain the signed
checksum, signature and verified image with the reviewed build inputs. If the
signing key changes, update the keyring through the trusted package repository or
an independently authenticated administrator process before retrying. The
builder's recorded input digest provides traceability; it does not authenticate
the publisher or replace this verification step.

Build `kbs-client` from the unmodified CoCo Trustee v0.22.0 checkout, using
upstream's Linux build prerequisites and Rust toolchain:

```sh
git clone --branch v0.22.0 https://github.com/confidential-containers/trustee.git /tmp/trustee
cargo build --locked --release --manifest-path /tmp/trustee/Cargo.toml   -p kbs-client --bin kbs-client --features tdx-attester,snp-attester
install -m 755 /tmp/trustee/target/release/kbs-client inputs/kbs-client
```

Keep the default crypto configuration: v0.22.0's optional `native-tls` feature
selects an OpenSSL RSA decryptor that does not support the RSA-OAEP-256 responses
used by this builder. Verify encrypted resource retrieval with the exact binary
before building a CVM.

For a GPU profile follow [GPU_BUILD.md](GPU_BUILD.md) to build the pinned NVAT
dependency and add `nvidia-attester`. Leave Trustee's source and dependency lockfile unchanged.
See [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) for the matching CoCo v0.23.0 backend.

The site operator supplies these trust inputs because they are deployment-specific
and cannot be downloaded from this repository:

- `inputs/kbs-ca.pem`: CA for the production KBS HTTPS endpoint.
- `inputs/as-public.pem`: public key used to verify Attestation Service tokens.
- `inputs/approved-tcb-references.json`: approved platform TCB reference values.
- A `site_acceptance` executable in the build host's root `PATH`.

Use reviewed, immutable copies of every input in production. The builder hashes
the base image, firmware, trust files, policy, runtime source, and final artifacts
into the CVM contract and manifest.

### Discover and approve TDX TCB references

The first build already requires `approved-tcb-references.json`. Discover its
candidate values in the temporary TD used for the host quote smoke test above,
before building a CVM. That TD needs no vault, KBS resource or approved CVM
references. Use the intended host's firmware/TDX module and the planned QEMU CPU
configuration; `xfam` describes the TD's enabled CPU state and must match the
eventual CVM. Do not fill the reference file with placeholder values to get past
build validation.

Copy the reviewed builder source into that temporary TD and install its Python
requirements as in §1. Inside the TD, change to its `image_builder` directory and
capture a local report with the same hardware adapter used by the CVM runtime:

```sh
sudo install -d -m 0700 /var/lib/cvm-tcb-discovery
sudo python3 - <<'PY_TCB'
import base64
from pathlib import Path

from cvm.common.evidence import verify_reference
from cvm.common.io import write_json
from cvm.common.measurements import measurements
from cvm.runtime.platforms import local_report

report, nonce = local_report("intel_tdx")
evidence = {
    "platform": "intel_tdx",
    "report": base64.b64encode(report).decode(),
    "nonce": base64.b64encode(nonce).decode(),
    "ccel": base64.b64encode(Path("/sys/firmware/acpi/tables/data/CCEL").read_bytes()).decode(),
    "measurements": measurements("intel_tdx", report),
}
verify_reference("intel_tdx", evidence)
write_json("/var/lib/cvm-tcb-discovery/reference-evidence.json", evidence)
PY_TCB
sudo ./cvmctl inspect-tcb /var/lib/cvm-tcb-discovery/reference-evidence.json
```

This uses `/dev/tdx_guest` and the guest's CCEL; run it inside the TD, not on the
host. The report file is mode 0600. `inspect-tcb` checks report structure, the
local nonce and measurement consistency, then prints `unapproved_candidate_tcb`:

| Field | TDREPORT bytes (zero-based, end excluded) | Reference encoding |
|---|---|---|
| `mr_seam` | `[280:328]` | List of 96-character lowercase hex strings |
| `tcb_svn` | `[264:280]` | List of 32-character lowercase hex strings |
| `xfam` | `[520:528]` | List of 16-character lowercase hex strings |

These are report bytes in hex; do not reverse `xfam` into integer notation.
The inspector does not authenticate a signed quote, replay CCEL, approve a TCB,
or modify Trustee. Keep reports and verifier results in restricted acceptance
records, outside OCI deliveries and public logs.

Have the platform administrator verify a fresh signed quote from the same TD
using the smoke-test workflow in §1 and the intended collateral channel. Check
signature/endorsements, challenge binding, collateral expiry and TCB status, and
compare the quote's `mr_seam`, `tcb_svn` and `xfam` with the candidates. With
Trustee v0.22.0, these fields are under `tdx.quote.body` in
[verified CPU evidence](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/deps/verifier/src/tdx/mod.rs);
`tdx.advisory_ids`, `tdx.tcb_status` and `tdx.collateral_expiration_status` come
from the verifier. An EAR exposes that evidence under
`submods.cpu0["ear.veraison.annotated-evidence"]`; validate the EAR's signature,
issuer, expiry, any configured audience and protocol binding before relying on it.
Decoding a JWT alone is not verification. See
[TCB channel selection](TDX_TROUBLESHOOTING.md#tcb-channel-selection).

`allowed_advisory_ids` is a separate administrator-approved list of advisory ID
strings from that verified appraisal, not a TDREPORT field or an automatically
accepted list. Use `[]` when no advisories are approved. Adding an advisory does
not bypass the CPU policy's `UpToDate` and unexpired-collateral requirements.
Keep KBS resource policy deny-all during discovery; no key release is needed.

After approval, write only `mr_seam`, `tcb_svn`, `xfam` and
`allowed_advisory_ids` as top-level keys in `inputs/approved-tcb-references.json`
for a CPU-only TDX profile. Omit the inspector's wrapper and `review_required`
text. Add the schema's SNP/GPU references only when those profiles are enabled.
The checked-in profile enables both CPU platforms by default; disable SNP for
a TDX-only profile. Unknown keys, including `_comment_*`, fail before building;
keep approval notes in a separate document. The exact schema is
[`cvm/common/references.py`](cvm/common/references.py).

Then build/finalize the real CVM and collect its own boot measurements. Do not
pass the temporary TD's evidence to `finalize --reference-evidence`: its
MRTD/RTMR values describe a different guest. On the trusted build host, inspect
the real bundle's private report with:

```sh
sudo ./cvmctl inspect-tcb \
  target/cvm_cpu-2026.09-r4/intel_tdx/reference-evidence.json
```

Compare its TCB fields with the approved input and complete signed-quote/CCEL
and deployment acceptance for that exact bundle before production approval.
Transfer the discovery records securely and stop the temporary TD afterward.
Changing approved references, firmware or other contract inputs requires a new
`profile_version`, fresh construction/finalization and approval, and the isolated
policy/reference storage described in [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md#6-install-references-and-policies).

## 2. Simple CVM Build

[config/cvm_profile.yml](config/cvm_profile.yml) is the default, complete Ubuntu
26.04 profile. Update the KBS URL and site-owned trust files, and assign a new
`profile_version` when any generic input changes. Then run one command on the
target SNP or TDX host:

```sh
sudo ./cvmctl build
```

No command-line parameters are required. The builder reads the default profile,
auto-detects the local platform, constructs the application-neutral CVM, boots
the exact result to collect reference measurements, invokes `site_acceptance`,
validates its exact-manifest report, and writes `approval.json`. A missing,
failed, or incomplete site acceptance report leaves the bundle unapproved.
`--acceptance-runner` is available only when a site uses a different executable.

Construction uses a disposable SSH key and a loopback-only forwarded port on the
trusted build host. The temporary port reservation is released before QEMU binds
it, and SSH host-key checking is disabled for the disposable guest. Run Stage 1 on
a host without untrusted local users; it carries no application vault keys. A
port collision fails the build, and production approval still requires the exact
bundle's acceptance report.

Registry transfers have no fixed CLI deadline because multi-disk artifacts can
take hours over slow links. Apply the site's transfer deadline externally when
needed. The trusted acceptance runner likewise owns the deadlines for its hardware
and soak tests; an unfinished runner never approves a bundle.

The main defaults are:

```yaml
profile_version: cpu-2026.09-r4
guest_release: '26.04'
gpu: none
base_image: ../inputs/ubuntu-26.04-server-cloudimg-amd64.img
build_firmware: /usr/share/ovmf/OVMF.fd
build_user: ubuntu
root_drive_size: 8
vcpus: 4
memory_gib: 8
# root_overlay_max_mib is optional and defaults to 4096 here.
kernel_version: 7.0.0-31-generic
kbs_url: https://kbs.example.org:8443
kbs_cert: ../inputs/kbs-ca.pem
as_public_key: ../inputs/as-public.pem
attestation_policy: attestation_policy.rego
reference_values: ../inputs/approved-tcb-references.json
platforms:
  amd_sev_snp:
    firmware: /usr/share/ovmf/OVMF.amdsev.fd
    kbs_client: ../inputs/kbs-client
  intel_tdx:
    firmware: ../inputs/OVMF.inteltdx.fd
    kbs_client: ../inputs/kbs-client
```

The checked-in profile also supplies the complete package pins, Trustee revision,
policy ID, quote-generation socket, and storage profile. Do not put an NVFlare
startup kit or `cc_params.yml` in this profile. NVFlare provisioning owns them
and may supply its output later as application content.

The reusable Stage 1 staging tree and OCI deliverable are:

```text
target/cvm_<profile_version>/
├── cvm_<profile_version>_<platform>.oci.tar
├── oci_artifacts.json
├── profile_set.json
└── <platform>/                       # build workspace
    ├── approval.json
    ├── cvm_manifest.json
    ├── verity_root.qcow2
    ├── OVMF.fd
    ├── vmlinuz
    └── initrd.img
```

The `.oci.tar` is the generic CVM deliverable. It is a standard OCI image layout
with artifact type `application/vnd.nvidia.cvm.bundle.v1`; it contains the platform
bundle and a single-platform `profile_set.json`. Build it once per
platform/profile version. The directory is retained as a build workspace. The
final application delivery is created by the Vault Build below.

`profile_set.json` is generated by the builder. It records `profile_version`
once at the top level, the shared runtime `contract`, and a `bundles` map keyed
by `amd_sev_snp` or `intel_tdx`. Each bundle entry has only `build_id` and
`manifest_sha256`. Its directory must have the same name as its platform key,
beside `profile_set.json`; there is no configurable `path`. Each CVM manifest
retains its own top-level `profile_version`, which must match the profile set.

## 3. Advanced CVM Build

Use the multi-step workflow to construct for several platforms or when the
construction machine is different from the target TEE host. Select a platform
and defer its measurements:

```sh
sudo ./cvmctl build config/cvm_profile.yml \
  -p amd_sev_snp --defer-measurements
```

The deferred call emits a pending CVM OCI artifact. Copy that `.oci.tar` to its
matching target host, verify and materialize it, then finalize it:

```sh
sudo ./cvmctl pull cvm_cpu-2026.09-r4_amd_sev_snp.oci.tar \
  --output /srv/cvm/cvm_cpu-2026.09-r4
sudo ./cvmctl finalize \
  /srv/cvm/cvm_cpu-2026.09-r4/amd_sev_snp
```

Run the site's acceptance matrix there. Then approve its exact report and install
the bundle's reference values and reusable resource policy:

```sh
sudo ./cvmctl admin approve \
  /srv/cvm/cvm_cpu-2026.09-r4/amd_sev_snp \
  /srv/cvm/acceptance-report.json

sudo ./cvmctl admin install \
  /srv/trustee/admin.json \
  /srv/cvm/cvm_cpu-2026.09-r4/amd_sev_snp
```

Finalization and approval regenerate the OCI artifact so it includes the final
manifest, profile set, evidence, resource policy and approval receipt. Repeat
construction and finalization once for each platform in the profile. Every
platform bundle must have the same generic contract. Use
`./cvmctl admin retire ADMIN_JSON BUILD_ID` to retire a bundle.

When the finalized platform artifacts return to the central Vault Build site,
materialize the first one and merge each additional platform. Merge verifies the
profile version, shared contract, platform entry and manifest digest before it
updates the combined `profile_set.json`:

```sh
./cvmctl pull cvm_cpu-2026.09-r4_intel_tdx.oci.tar \
  --output target/final_cvm_cpu-2026.09-r4
./cvmctl pull cvm_cpu-2026.09-r4_amd_sev_snp.oci.tar \
  --output target/final_cvm_cpu-2026.09-r4 --merge
```

Set `cvm_image: ../target/final_cvm_cpu-2026.09-r4` in
`config/vault_build.yml` to use that aggregated folder.

## 4. Vault Build

Configure the shared key endpoint once in [cvm_project.yml](cvm_project.yml)
at your project root. The checked-in file uses paths under the builder's `inputs/`:

```yaml
trustee:
  url: https://trustee.example.org:8443
  ca: ./inputs/kbs-ca.pem
  admin_token_file: ./inputs/kbs-resource-token.jwt
```

Use your actual HTTPS endpoint and scoped resource-administration token from
[TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md#5-use-native-trustee-resource-administration).
Credential paths resolve relative to `cvm_project.yml`. Each vault build searches
from its build YAML directory upward and uses the nearest `cvm_project.yml`.
It does not search from the shell's working directory or merge ancestor files.
A missing or invalid project configuration fails before CVM retrieval or key creation.
Do not put `trustee` in `vault_build.yml`; per-build overrides are rejected.

For generated YAML outside the project, select the shared file explicitly:

```sh
sudo ./cvmctl vault /tmp/site-inputs/vault_build.yml \
  --project-config /srv/project/cvm_project.yml
```

Relative `--project-config` paths resolve from the shell's working directory.
This option selects one complete project file; a missing explicit file never
falls back to discovery. Plaintext `--dev` builds skip project discovery and
reject `--project-config`. Candidate builds still require the project Trustee configuration.

The container can run any Linux amd64 application. Pull the selected image, print
its immutable `image_id`, and save it as an archive:

```sh
image=my-application:1.0
docker pull "$image"
docker image inspect --format '{{.Id}}' "$image"
docker save "$image" -o inputs/application.tar
```

Copy the exact value printed by `docker image inspect` into `image_id` in
[config/vault_build.yml](config/vault_build.yml):

```yaml
cvm_image: ../target/cvm_cpu-2026.09-r4
docker_archive: ../inputs/application.tar
image_id: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
# Optional: omit to use every platform available in cvm_image.
# platforms: [intel_tdx]
requires_gpu: false
vault_drive_size: 8
applog_drive_size: 1
user_config_drive_size: 1
user_data_drive_size: 1
allowed_ports: [8080]
allowed_out_ports: [443, 8443]
# Optional encrypted application payload and clear read-only input trees:
# application_files: ../inputs/application-files
# user_config: ../inputs/user-config
# user_data: ../inputs/user-data
container:
  ports: [{host: 8080, container: 8080}]
  env: {}
  volumes: []
  tee_device: false
services: []
hosts_entries: {}
```

Optional `services` entries point to unit files named `app_<name>.service`,
using lowercase letters, digits, and underscores, such as `app_helper.service`.
The builder supplies their bootstrap dependencies and PID 1 failure actions.

`cvm_image` accepts a local folder containing `profile_set.json` and its platform
subdirectories, or a generic CVM OCI registry reference pinned by manifest digest:

```yaml
cvm_image: registry.example.org/cvm/cpu-2026.09-r4@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`oci://` and `https://` prefixes are also accepted. Use the actual digest printed
by `./cvmctl publish`. Registry retrieval requires ORAS and uses HTTPS by
default. The builder downloads and verifies the generic CVM, requires its
approval receipt, and keeps the retrieved files until they have been copied into
the delivery. A registry artifact contains one platform; a local folder may
contain several merged platform bundles. For offline use, materialize the CVM
`.oci.tar` with `./cvmctl pull` first and use its output folder.

`deployment_id` is generated automatically for every invocation; do not put it
in the YAML. The builder prints the ID and resulting artifact paths, and records
the ID in `vault_set.json` and each vault manifest. `--output /absolute/folder`
can select a new output folder without changing the generated artifact identity.

The four drive sizes shown above are the defaults. `platforms` is optional and
defaults to every platform available in `cvm_image`. Set it only to select a
subset; each requested platform must be available. `application_files` is optional encrypted content
under `/vault/application`. `user_config` and `user_data` are optional clear,
host-readable inputs mounted read-only in the guest and container. `/applog` is
the only writable clear sidecar and is an output-only channel for logs the
operator must read without a KBS key. Put secrets and confidential logs in
`/vault`. For NVFlare, put the startup kit produced by NVFlare provisioning in
`application_files`.

Build a fresh vault for this application release and site:

```sh
sudo ./cvmctl vault config/vault_build.yml
```

For the isolated HTTP lab registry, use a digest reference such as
`registry.example.org:5000/cvm/cpu@sha256:<manifest-digest>` and explicitly opt in:

```sh
sudo ./cvmctl vault config/vault_build.yml --plain-http
```

### Use a GPU inside the container

Use a generic CVM profile with `gpu: nvidia_cc`, the required `gpu_count` from 1
through 8, a reviewed `gpu_policy`, a trusted `gpu_attestation_url`, the pinned
`gpu_attestation_library`, its `gpu_attestation_provenance`, authenticated
`gpu_apt_repositories`, and exact `gpu_packages` pins for the NVIDIA guest driver
and NVIDIA Container Toolkit. [GPU_BUILD.md](GPU_BUILD.md) supplies the concrete
repository/keyring inputs, package pins, and NVAT build recipe for a clean base image.
Set `requires_gpu: true` in `vault_build.yml`. The builder rejects a GPU
application paired with a CPU-only profile and rejects a GPU profile paired with
an application that does not request the GPU.

Point `gpu_policy` at `config/gpu_policy.json` unless a site needs stricter
rules. The upstream client collects raw evidence using NVIDIA's `libnvat` SDK bindings
and includes it in the same KBS transaction as the CPU quote.
Trustee calls NRAS and authenticates its signed EAT; the generated AS GPU policy
compares every nested value in `required-claims` and driver/VBIOS versions against
RVPS approvals before KBS can release the vault key. Stage 1 validates
that the policy requires secure boot, disabled debug mode, successful
measurements, valid report/RIM certificate and OCSP state, verified signatures,
matching driver/VBIOS RIMs and no VBIOS-index conflict. Adding constraints is
supported; removing these minimum constraints is rejected.

Current NRAS responses can omit the driver and VBIOS RIM schema-validation
fields. The policy lists those two fields under `claims-if-present`: a returned
false or malformed value denies key release. RIM signature, certificate,
version and measurement checks remain mandatory in `required-claims`.

The default expects `inputs/libnvat.so.1` plus `inputs/nvat_build.json`.
Trustee v0.22.0's Cargo.lock pins the NVAT 2026.03.02 source, which builds library
version 1.2.0 with soname 1. Stage 1 checks the source revision, reviewed libxml2
compatibility patch, build environment and binary digest in the provenance
record. Evidence collection and verification use CoCo's code.

For a smaller guest, `gpu_packages` may pin a precompiled
`linux-modules-nvidia-*-<kernel>` package plus its matching
`libnvidia-compute-*`, compute utilities and NVIDIA Container Toolkit packages.
This avoids desktop libraries and DKMS; pin the kernel module and every user-space
package to one compatible driver release. A pinned `nvidia-driver-*-open` meta
package is also accepted when a site intentionally validates the complete
dependency set.

Use NVIDIA's official sources to select compatible, pinned inputs:

- Driver packages: <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- GPU attestation CLI source: <https://github.com/NVIDIA/attestation-sdk>
- GPU attestation CLI command reference: <https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html>
- GPU claim schema: <https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html>

No runtime GPU argument is needed in the usual case. `launch_cvm.sh` detects
exactly `gpu_count` NVIDIA display GPUs, binds every function in each isolated
PCI slot to `vfio-pci`, and passes all selected slots to QEMU. Each GPU gets a
dedicated PCIe root port with a 256 GiB 64-bit prefetchable MMIO window; this
accommodates current data-center devices such as an H800 with a 128 GiB BAR.
TDX uses one IOMMUFD backend for the assigned GPUs, so the runtime host's QEMU
must provide the `iommufd` object. GPU profiles also measure
`pci=realloc,nocrs` into the guest command line so Linux allocates the GPU's
large PCI BARs. Inside the CVM, the measured guest configures
`nvidia-container-runtime`; the application starts with the equivalent of:

```sh
docker run --gpus all APPLICATION_IMAGE
```

The application image must contain CUDA/application user-space components that
are compatible with the pinned guest driver. For explicit placement, repeat `launch_cvm.sh --gpu PCI_ADDRESS` once for every GPU required by `gpu_count`.

## 5. Deliverables

Vault Build prints its generated deployment ID and staging folder. For example,
an ID of `0123456789ab4def8123456789abcdef` produces:

```text
target/vault_0123456789ab4def8123456789abcdef/
```

Each selected platform has one deployable OCI image-layout tar:

```text
target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_amd_sev_snp.oci.tar
target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar
```

The platform OCI artifact is the delivery to copy to its target site or publish
to a registry. Its artifact type is
`application/vnd.nvidia.cvm.delivery.v1`. One deterministic layer holds the
already-approved generic CVM bytes so a registry can deduplicate it across
applications. A second layer holds the sealed application vault, clear directional sidecars,
delivery manifest, launch scripts and host launcher modules. Materializing the
artifact produces the complete `cvm_bundle/` layout; no separate Stage 1 download
is required. The adjacent platform directory contains the same staged contents
for inspection and is not the distributed artifact.

`oci_artifacts.json` records the archive SHA-256 and OCI manifest digest. For an
offline delivery, copy the `.oci.tar` and publish its `archive_sha256` through an
authenticated channel:

```sh
sha256sum target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar
```

For a registry delivery, publish the exact OCI layout with ORAS:

```sh
./cvmctl publish \
  target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar \
  registry.example.org/cvm/my-app-site1:intel-tdx-1.0
```

An isolated HTTP test registry requires an explicit opt-in:

```sh
./cvmctl publish \
  target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar \
  registry.example.org:5000/cvm/my-app-site1:intel-tdx-1.0 --plain-http
```

The wrapper prints the immutable `registry/repository@sha256:...` reference. Sign
that digest for production and give the digest reference, never only the mutable
tag, to runtime operators:

```sh
cosign sign --key /secure/release-signing.key \
  registry.example.org/cvm/my-app-site1@sha256:OCI_MANIFEST_DIGEST
```

The same `./cvmctl publish` command can publish a Stage 1 CVM `.oci.tar` when
another build site needs that reusable bundle.

Rebuild the vault when the application image, application files, site, or
configuration changes. Reuse the approved generic CVM bundle.

Continue with [USER_GUIDE.md](USER_GUIDE.md).

### Trusted NFS configuration and application write access

Configure an NFS input in the encrypted `application.json`, not a clear sidecar:

```json
"nfs_mount": {"server": "files.example.org", "export": "/datasets", "security": "krb5p"}
```

The guest mounts `/user_data/mnt` with `ro,nosuid,nodev,noexec,sec=krb5p`.
Provision the site's Kerberos configuration and credentials through protected
vault inputs and reviewed guest services; permit the required KDC/NFS egress.
There is no unauthenticated fallback. Legacy `/user_data/ext_mount.conf` is
rejected because host-controlled bytes must not select a root kernel NFS peer.

Containers receive `/vault/application` read-only; only its `runtime/` and `data/`
subdirectories are writable. Place application state and confidential logs there.
Optional volume mappings cannot expose `/vault/config`, `/vault/services`, Docker
storage or other vault control files. Vault services remain trusted code admitted
by the builder; their executable must resolve inside the immutable application
payload, outside those writable subdirectories. This is not a sandbox for a
malicious root guest service.

Raw CPU reference reports stay in mode-0600 build/acceptance records and are
excluded from OCI archives. Normal vault boots do not print reports to the serial
console. Public manifests include only required launch measurements and omit
plaintext application content hashes. Retain acceptance logs under the site's
restricted evidence policy.
