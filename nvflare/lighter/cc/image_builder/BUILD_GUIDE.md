# Build a CVM bundle and application vault

The generic CVM contains Ubuntu, the container runtime, attestation code, and the
vault unlock path. Build it once for each platform and profile version, then reuse
it for every compatible application. Build a new vault for every application
release and site.

Set up the attestation and key backend first using
[TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md). It includes service configuration, test
certificates, reference-value installation, and the `key_service` credentials
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

The official Ubuntu 26.04 `ovmf` package provides the default plain, SNP, and TDX
firmware paths used by `config/cvm_profile.yml`. The package and its component
packages are listed at <https://packages.ubuntu.com/resolute/ovmf>. Verify the
files after installation:

```sh
test -r /usr/share/ovmf/OVMF.fd
test -r /usr/share/ovmf/OVMF.amdsev.fd
test -r /usr/share/ovmf/OVMF.inteltdx.ms.fd
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
  cvm_image_name=ubuntu-26.04-server-cloudimg-amd64.img
  cvm_image_url=https://cloud-images.ubuntu.com/releases/26.04/release
  curl -fL "$cvm_image_url/SHA256SUMS" -o SHA256SUMS
  curl -fL "$cvm_image_url/SHA256SUMS.gpg" -o SHA256SUMS.gpg
  gpgv --keyring /usr/share/keyrings/ubuntu-cloudimage-keyring.gpg \
    SHA256SUMS.gpg SHA256SUMS
  awk -v image="$cvm_image_name" '
    $2 == image || $2 == "*" image { print; found++ }
    END { if (found != 1) exit 1 }
  ' SHA256SUMS > "$cvm_image_name.sha256"
  curl -fL "$cvm_image_url/$cvm_image_name" -o "$cvm_image_name"
  sha256sum --check --strict "$cvm_image_name.sha256"
)
```

Proceed with construction only after this block succeeds. Retain the signed
checksum, signature and verified image with the reviewed build inputs. If the
signing key changes, update the keyring through the trusted package repository or
an independently authenticated administrator process before retrying. The
builder's recorded input digest provides traceability; it does not authenticate
the publisher or replace this verification step.

Build the pinned `kbs-client` from the official Trustee repository. The reviewed
source revision is
<https://github.com/confidential-containers/trustee/tree/a2570329cc33daf9ca16370a1948b5379bb17fbe>:

```sh
git clone https://github.com/confidential-containers/trustee.git /tmp/trustee
cd /tmp/trustee
git checkout a2570329cc33daf9ca16370a1948b5379bb17fbe
cd -
python3 scripts/patch_trustee.py /tmp/trustee
cargo build --locked --release \
  --manifest-path /tmp/trustee/Cargo.toml \
  -p kbs-client --bin kbs-client --no-default-features \
  --features tdx-attester,snp-attester,kbs_protocol/background_check,kbs_protocol/passport,kbs_protocol/rust-crypto
install -m 755 /tmp/trustee/target/release/kbs-client inputs/kbs-client
```

The site operator supplies these trust inputs because they are deployment-specific
and cannot be downloaded from this repository:

- `inputs/kbs-ca.pem`: CA for the production KBS HTTPS endpoint.
- `inputs/as-public.pem`: public key used to verify Attestation Service tokens.
- `inputs/approved-tcb-references.json`: approved platform TCB reference values.
- A `site_acceptance` executable in the build host's root `PATH`.

Use reviewed, immutable copies of every input in production. The builder hashes
the base image, firmware, trust files, policy, runtime source, and final artifacts
into the CVM contract and manifest.

## 2. Simple CVM Build

[config/cvm_profile.yml](config/cvm_profile.yml) is the default, complete Ubuntu
26.04 profile. Update the KBS URL and site-owned trust files, and assign a new
`profile_version` when any generic input changes. Then run one command on the
target SNP or TDX host:

```sh
sudo ./cvm_build.sh
```

No command-line parameters are required. The builder reads the default profile,
auto-detects the local platform, constructs the application-neutral CVM, boots
the exact result to collect reference measurements, invokes `site_acceptance`,
validates its exact-manifest report, and writes `approval.json`. A missing,
failed, or incomplete site acceptance report leaves the bundle unapproved.
`--acceptance-runner` is available only when a site uses a different executable.

The main defaults are:

```yaml
profile_version: cpu-2026.09
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
    firmware: /usr/share/ovmf/OVMF.inteltdx.ms.fd
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
sudo ./cvm_build.sh config/cvm_profile.yml \
  -p amd_sev_snp --defer-measurements
```

The deferred call emits a pending CVM OCI artifact. Copy that `.oci.tar` to its
matching target host, verify and materialize it, then finalize it:

```sh
sudo scripts/cvm_pull cvm_cpu-2026.09_amd_sev_snp.oci.tar \
  --output /srv/cvm/cvm_cpu-2026.09
sudo scripts/cvm_finalize \
  /srv/cvm/cvm_cpu-2026.09/amd_sev_snp
```

Run the site's acceptance matrix there. Then approve its exact report and install
the bundle's reference values and reusable resource policy:

```sh
sudo scripts/admin_approve \
  /srv/cvm/cvm_cpu-2026.09/amd_sev_snp \
  /srv/cvm/acceptance-report.json

sudo scripts/admin_install \
  /srv/trustee/admin.json \
  /srv/cvm/cvm_cpu-2026.09/amd_sev_snp
```

Finalization and approval regenerate the OCI artifact so it includes the final
manifest, profile set, evidence, resource policy and approval receipt. Repeat
construction and finalization once for each platform in the profile. Every
platform bundle must have the same generic contract. Use
`scripts/admin_retire ADMIN_JSON BUILD_ID` to retire a bundle.

When the finalized platform artifacts return to the central Vault Build site,
materialize the first one and merge each additional platform. Merge verifies the
profile version, shared contract, platform entry and manifest digest before it
updates the combined `profile_set.json`:

```sh
scripts/cvm_pull cvm_cpu-2026.09_intel_tdx.oci.tar \
  --output target/final_cvm_cpu-2026.09
scripts/cvm_pull cvm_cpu-2026.09_amd_sev_snp.oci.tar \
  --output target/final_cvm_cpu-2026.09 --merge
```

Set `cvm_image: ../target/final_cvm_cpu-2026.09` in
`config/vault_build.yml` to use that aggregated folder.

## 4. Vault Build

Configure the shared key endpoint once in [cvm_project.yml](cvm_project.yml)
at your project root. The checked-in file uses paths under the builder's `inputs/`:

```yaml
key_service:
  url: https://key-service.example.org:9443
  ca: ./inputs/key-service-ca.pem
  cert: ./inputs/builder-client.pem
  key: ./inputs/builder-client.key
```

Use your actual HTTPS endpoint and mTLS credential files from
[TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md#9-upload-keys-through-vault-build).
Credential paths resolve relative to `cvm_project.yml`. Each vault build searches
from its build YAML directory upward and uses the nearest `cvm_project.yml`.
It does not search from the shell's working directory or merge ancestor files.
A missing or invalid project configuration fails before CVM retrieval or key creation.
Do not put `key_service` in `vault_build.yml`; per-build overrides are rejected.

For generated YAML outside the project, select the shared file explicitly:

```sh
sudo ./vault_build.sh /tmp/site-inputs/vault_build.yml \
  --project-config /srv/project/cvm_project.yml
```

Relative `--project-config` paths resolve from the shell's working directory.
This option selects one complete project file; a missing explicit file never
falls back to discovery. Plaintext `--dev` builds skip project discovery and
reject `--project-config`. Candidate builds still require the project key service.

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
cvm_image: ../target/cvm_cpu-2026.09
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

`cvm_image` accepts a local folder containing `profile_set.json` and its platform
subdirectories, or a generic CVM OCI registry reference pinned by manifest digest:

```yaml
cvm_image: registry.example.org/cvm/cpu-2026.09@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`oci://` and `https://` prefixes are also accepted. Use the actual digest printed
by `scripts/cvm_publish`. Registry retrieval requires ORAS and uses HTTPS by
default. The builder downloads and verifies the generic CVM, requires its
approval receipt, and keeps the retrieved files until they have been copied into
the delivery. A registry artifact contains one platform; a local folder may
contain several merged platform bundles. For offline use, materialize the CVM
`.oci.tar` with `scripts/cvm_pull` first and use its output folder.

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
sudo ./vault_build.sh config/vault_build.yml
```

For the isolated HTTP lab registry, use a digest reference such as
`registry.example.org:5000/cvm/cpu@sha256:<manifest-digest>` and explicitly opt in:

```sh
sudo ./vault_build.sh config/vault_build.yml --plain-http
```

### Use a GPU inside the container

Use a generic CVM profile with `gpu: nvidia_cc`, the required `gpu_count` from 1
through 8, a reviewed `gpu_policy`, a trusted `gpu_attestation_url`, the pinned
`gpu_attestation_binary` and `gpu_attestation_library`, and exact `gpu_packages`
pins for the NVIDIA guest driver and NVIDIA Container Toolkit.
Set `requires_gpu: true` in `vault_build.yml`. The builder rejects a GPU
application paired with a CPU-only profile and rejects a GPU profile paired with
an application that does not request the GPU.

Point `gpu_policy` at `config/gpu_policy.json` unless a site needs stricter
rules. The measured root runs NVIDIA's C++ `nvattest` CLI, requires a successful
remote appraisal with signed EAT evidence, checks the fresh nonce and exact GPU
count, and compares every nested value in `required-claims`. Stage 1 validates
that the policy requires secure boot, disabled debug mode, successful
measurements, valid report/RIM certificate and OCSP state, verified signatures,
matching driver/VBIOS RIMs and no VBIOS-index conflict. Adding constraints is
supported; removing these minimum constraints is rejected.

Current NRAS responses can omit the driver and VBIOS RIM schema-validation
fields. The policy lists those two fields under `claims-if-present`: a returned
false or malformed value still denies startup. RIM signature, certificate,
version and measurement checks remain mandatory in `required-claims`.

The checked-in defaults expect `inputs/nvattest` and
`inputs/libnvat.so.1.2.2`. Obtain the source from NVIDIA's current C++
attestation SDK, check out a reviewed release, build the CLI for the selected
guest release, and record its source revision and file digests in release
provenance. The September 2026 Ubuntu 26.04 validation used SDK commit
`9d12801cea8a198ea0f29640dfaf8a4017c841c5` (NVAT 1.2.2). NVIDIA's prebuilt
packages currently target Ubuntu 22.04 and 24.04, so they are not substituted
into an Ubuntu 26.04 profile.

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
scripts/cvm_publish \
  target/vault_0123456789ab4def8123456789abcdef/vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar \
  registry.example.org/cvm/my-app-site1:intel-tdx-1.0
```

An isolated HTTP test registry requires an explicit opt-in:

```sh
scripts/cvm_publish \
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

The same `scripts/cvm_publish` command can publish a Stage 1 CVM `.oci.tar` when
another build site needs that reusable bundle.

Rebuild the vault when the application image, application files, site, or
configuration changes. Reuse the approved generic CVM bundle.

Continue with [USER_GUIDE.md](USER_GUIDE.md).
