# CVM Builder

Build a reusable, application-free Ubuntu 26.04 confidential VM and a separate
authenticated vault for each application release and site. Docker workloads are
generic: the image's entrypoint and command are preserved by default. NVFlare is
the main use case; its startup kit and `cc_params.yml` belong to NVFlare
provisioning, which supplies ordinary application files to this builder.

The security contract and acceptance requirements are in [DESIGN.md](DESIGN.md).
Production delivery requires an approved generic bundle; `--candidate` is limited
to explicitly named `test-` profiles and never creates production approval.
Both reusable CVM bundles and final CVM-plus-vault deliveries are emitted as OCI
image-layout tar files. They can be moved offline or published to a registry.
Use `scripts/cvm_publish` for registry publication and `scripts/cvm_pull` to
materialize either a local OCI tar or an immutable registry digest.

This builder is part of NVFlare at `nvflare/lighter/cc/image_builder`.
Run the commands below from this directory in a source checkout. NVFlare
provisioning invokes `vault_build.sh` through its `cvm_vault` configuration; see
the [provisioning example](../../../../examples/advanced/cc_provision/cvm_builder/README.md).

For operational instructions, see:

- [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md): configure the existing CoCo Trustee,
  enable approved CVM bundles, and manage vault keys.
- [BUILD_GUIDE.md](BUILD_GUIDE.md): build the reusable CVM bundle and a fresh
  vault for each application release and site.
- [USER_GUIDE.md](USER_GUIDE.md): verify, launch, operate, and stop a delivered CVM.
- [TDX_TROUBLESHOOTING.md](TDX_TROUBLESHOOTING.md): diagnose Intel quote-generation
  failures, recover QGS after PCK cache expiry, and select the Trustee TCB channel.

## Inputs and prerequisites

Run builds as root on an x86-64 Linux host with KVM, swap disabled, and sufficient
locked memory. Shared testing targets Ubuntu 26.04. The construction VM itself
runs without confidential-computing mode, so `-p` allows cross-platform builds.
Reference collection and runtime launch require the selected SNP or TDX hardware.
Disable piped core collectors on build and Trustee hosts before handling keys;
the tools check this explicitly. Linux ignores `RLIMIT_CORE` for pipe handlers.
See [core(5)](https://man7.org/linux/man-pages/man5/core.5.html). The measured guest
masks crash collectors and sets `kernel.core_pattern=/dev/null` with
`kernel.core_uses_pid=0`, in addition to process-level dump protection.

Install Python 3, PyYAML, cryptography, QEMU system/utilities, cloud-image-utils,
libguestfs-tools, cryptsetup-bin, e2fsprogs, tar, util-linux, and OpenSSH. Storage
requires kernel dm-crypt, dm-integrity and NBD support. TDX requires an operational
QGS and TDVF supporting direct measured kernel boot. SNP requires SNP-enabled
KVM and compatible OVMF. GPU profiles additionally need pinned driver/toolkit
packages, the upstream NVIDIA attester’s pinned `libnvat` library, and an approved GPU
claim policy.

Install the builder requirements with pip or uv:

```sh
python3 -m pip install -r requirements.txt

# Or:
uv pip install -r requirements.txt
```

All input paths are relative to their YAML file. Production inputs are supplied
explicitly; the builder does not fetch firmware, private keys, or mutable
container tags.
Use [config/cvm_profile.yml](config/cvm_profile.yml) and
[config/vault_build.yml](config/vault_build.yml) as configuration templates.
They provide concrete Ubuntu 26.04 defaults, conventional locations, and sample
values. A bare `sudo ./cvm_build.sh` reads that profile and auto-detects the one
local target platform; no platform argument is required.

## Stage 1: build once per platform and profile version

```sh
# One target: construct, finalize, run acceptance, and approve in one call.
sudo ./cvm_build.sh

# Advanced: construct for another platform and finalize there.
sudo ./cvm_build.sh config/cvm_profile.yml -p amd_sev_snp --defer-measurements
sudo scripts/cvm_finalize target/cvm_cpu-2026.09-r4/amd_sev_snp
```

The builder sends a fixed, source-hashed payload to a plain construction VM. A
purpose-built Python provisioner installs the exact package pins, configures and
hardens the root, exports the boot artifacts, removes its temporary SSH access,
and shuts the VM down. It requires no configuration-management runtime. The
builder then seals the generic root with dm-verity and collects public hardware
reference evidence. Its initramfs only activates verity and a temporary overlay;
the KBS client and public trust settings reside in the verified root. The measured
command line disables systemd's separate GPT verity discovery because the
initramfs has already opened the root mapping. `root_overlay_max_mib` caps the
writable RAM-backed root layer; when omitted, the builder sets it to half of the
configured guest memory. The limit is part of the measured generic profile.

The output directory contains one workspace subdirectory per platform, a shared
`profile_set.json`, and one `cvm_<profile>_<platform>.oci.tar` deliverable per
platform. Each bundle contains `verity_root.qcow2`, `OVMF.fd`, `vmlinuz`,
`initrd.img`, `cvm_manifest.json`, public reference evidence, reference values, an
AS policy, and its reusable resource-policy rule. The root includes pinned chrony
and refuses all attestation until its clock meets the measured synchronization
gate. A deferred build has
`cvm_manifest.pending.json` until finalization. Construction logs are retained on
failure. Completed bundles cannot be overwritten; changes to the generic
runtime, dependencies, policy or trust settings require a new profile version.
The hashed `launch_cvm.sh.tmpl` and `shutdown_cvm.sh.tmpl` supply each vault's
runtime wrappers.

Both platform bundles must report an identical vault-facing contract. Firmware,
CPU model, attester and launch measurements are platform-specific. The provisioned
runtime source is snapshotted at build start to prevent edits during a build from
changing its declared source hash.

The simple workflow invokes `site_acceptance` from `PATH` with the finalized
bundle directory and the path where it must write the report. A site-specific
command can be selected with `--acceptance-runner`. This keeps construction,
finalization, acceptance, and approval in one call while preserving exact-manifest
evidence binding. Hardware reference collection alone does **not** approve a bundle. A trusted
operator must validate the signed quote, replay TDX CCEL where applicable, verify
the TCB references, and complete the design's acceptance matrix. An acceptance
report must identify the exact manifest hash and contain successful, hashed
evidence for every check returned by `cvm.artifacts.bundle.required_acceptance_checks`
for that platform and CPU/GPU profile:

```sh
sudo scripts/admin_approve /path/to/bundle /path/to/acceptance-report.json
```

Keep approval receipts and bundle artifacts under trusted administrative control.
The receipt records operator approval; it is not a substitute for those tests.

## Trustee administration

Use unmodified **CoCo Trustee v0.22.0**, paired with **CoCo v0.23.0**.
The profile pins upstream commit `512fed65642015b849f38fb13bfdec7806639987`;
there is no custom Rust verifier, attester, or Trustee patch to apply.
Use the same upstream distribution and image digest as your CoCo deployment.

[TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) contains the complete setup, including the
upstream [kbs.json](trustee/kbs.json) configuration, immutable default CPU/GPU
policies, RVPS references and expiry, role-based administrative ACLs, and native
resource uploads. `scripts/trustee_provenance` records a clean
release checkout and binary digest. The upstream client uses the `default` AS
policy; policy content digests and profile versions identify approved revisions.

CVM-specific authorization remains in Rego and deployment configuration. The
resource policy requires a fresh, favorable CPU appraisal and, for GPU profiles,
exactly the expected favorable NVIDIA GPU appraisals before key release.
Vault builds use a scoped bearer token for native KBS resource POST. Policy
publishing remains a separate administrative role. CVM Builder runs no custom
key server and ships no Trustee systemd services; CoCo manages the backend.
Native uploads may overwrite, and native deletion has no permanent tombstone.
Fence active builds before deletion and preserve revocations during backup restore.

Administration verifies the installed upstream revision, binary digest, policy
content hashes, approved reference values and deployment acceptance before
publishing bundle resource rules:

```sh
sudo scripts/admin_install admin.json /path/to/approved/bundle
sudo scripts/admin_retire admin.json cvm-BUNDLE_ID
```

There is no per-vault measurement history. Reference values and keys live in
Trustee storage; bundle retirement records stay in the publisher's admin state. Rebuild and
reapprove generic bundles when migrating from the earlier backend.

## Stage 2: rebuild for each application release and site

Create a Docker save archive and record the image's immutable ID. Configure
`docker_archive`, `image_id`, application files, ports, optional mounts/environment,
and `cvm_image` in `config/vault_build.yml`. Configure the existing Trustee endpoint and scoped
resource token once in [cvm_project.yml](cvm_project.yml). Vault Build finds
the nearest project file above the build YAML; credential paths are relative to
that project file. Use `--project-config /path/cvm_project.yml` for build inputs
staged elsewhere. Per-build `trustee` fields are rejected:

```sh
docker image inspect --format '{{.Id}}' my-application:release
docker save my-application:release -o application.tar
sudo ./vault_build.sh config/vault_build.yml
```

Application content is populated inside authenticated storage once, then copied
directly between encrypted mappings. Each selected platform receives an independent
authenticated LUKS2 vault with its own UUID and 64-byte secret. It also receives
clear directional sidecars: writable CVM output in `applog`, and read-only
operator inputs in `user_config` and `user_data`. Content
digests agree across those copies. No plaintext vault image, key file, or writable
shared backing image is produced. Only vault-key creation reaches the backend;
Stage 2 does not change AS policies, reference values, or bundle rules.

Set `cvm_image` to the generic CVM folder containing `profile_set.json`, or to its
immutable `registry/repository@sha256:...` reference. Registry input is retrieved
through ORAS and verified before use. Omit `platforms` to seal every available
platform, or supply a subset. Do not supply `deployment_id`; each invocation
generates a UUID-based ID and records it in `vault_set.json` and the manifests.

Outputs are under `target/vault_<deployment_id>/<platform>/`, using that generated ID.
The CLI prints the staging folder and each artifact path. Each platform's
`vault_<deployment_id>_<platform>.oci.tar` is self-contained: it includes the complete already-built
generic bundle under `cvm_bundle/`, plus `vault_manifest.json`, the sealed vault
and sidecars. Copying those generic bytes does not rebuild Stage 1, and the
runtime host does not need another bundle download.
The public delivery manifest contains the bundle identity and header binding. Private application
configuration and the Docker archive reside inside `vault.qcow2`. Keep the output
private while building; only distribute completed deliveries. On an uncertain
upload, retain `build_failure.json`, `provisioning.json`, and the encrypted image.
An administrator must resolve/revoke the recorded resource before rebuilding;
builder credentials intentionally cannot revoke a possibly active key.

Each completed copy is packaged as a two-layer OCI artifact: a reusable generic
CVM layer plus a deployment-specific vault/runtime layer. Digests are recorded in
`oci_artifacts.json`. OCI artifacts exclude administrative provisioning records
and Python bytecode caches.

```sh
# Use the staging folder printed by vault_build.sh.
cd target/vault_0123456789ab4def8123456789abcdef/intel_tdx
sudo ./launch_cvm.sh
# From another shell:
sudo ./shutdown_cvm.sh
```

The launcher verifies the embedded `cvm_bundle/` by platform and build id. GPU
profiles detect exactly the configured `gpu_count`, bind every function in each GPU slot to VFIO, pass all selected slots to QEMU, and restore the host drivers after exit. The measured guest configures the
NVIDIA container runtime and starts a GPU application with `docker run --gpus all`;
repeat `--gpu PCI_ADDRESS` exactly `gpu_count` times for explicit placement. The launcher pins measured inputs
and uses exclusive locking on the actual vault file. One runtime vault file belongs to one CVM at a time. File copies may be
launched independently; a copy retains the original identity and revocation scope.
Shut the CVM down before copying its vault or `applog` disk.

Disk roles use explicit SCSI serials (`cvm-root`, `cvm-applog`, `cvm-user-config`,
`cvm-user-data`, `cvm-vault`). The guest resolves their `/dev/disk/by-id/` paths;
asynchronous Linux disk letters and SCSI target order do not select mounts.

The guest hashes the complete 16 MiB LUKS header and compares it with the local
SNP host-data or all 48 bytes of TDX MRCONFIGID before contacting KBS. The checked
header is sealed in memory and reused for unlocking. The KBS secret opens only
the vault. The integrity monitor starts before the full authenticated vault scan
and filesystem mount; the clear sidecars are mounted afterward with fixed access modes. Docker and containerd
stores live in the vault; first-load completion is recorded only after the expected
image exists. A failed integrity monitor, failed periodic appraisal, or revoked key
stops the workload and powers off the guest.
Bootstrap and periodic appraisal write allowlisted metadata to the journal and
`/applog/attestation.log`; token, key and application values are never audit fields.

The container receives `/vault/application` read-only, its `runtime/` and `data/`
subdirectories writable, `/applog`, `/user_config` (read-only), `/user_data`
(read-only), and `/host/bin` (read-only). Additional mounts and command overrides
are optional. TEE-device access is opt-in and platform-neutral in the application
configuration. `/applog` is a clear output-only channel so an operator can read
logs without a KBS key. `/user_config` and `/user_data` are clear, host-readable,
untrusted inputs. QEMU opens both input disks read-only, the guest mounts them
`ro,noload,nosuid,nodev,noexec`, and Docker bind-mounts them read-only. The builder
rejects private-key filenames, containers, PEM content and symlinks in both input
trees. Put application state and confidential logs in `/vault/application/data`.
Optional NFS input uses authenticated `nfs_mount` configuration in the encrypted
application JSON and Kerberos `krb5p`; clear `ext_mount.conf` is rejected. See
[BUILD_GUIDE.md](BUILD_GUIDE.md#trusted-nfs-configuration-and-application-write-access).

Each CPU appraisal transaction has one 60-second budget across attestation and
resource retrieval. GPU profiles use a single 240-second composite CPU/GPU
transaction: KBS verifies NVIDIA evidence and requires the exact configured GPU
count before key release. CUDA readiness is enabled only afterward. The periodic
service has a 300-second outer deadline; hardware timing acceptance is pending. Silent packet loss,
negative appraisal, GPU failure, integrity failure or lost clock synchronization
stops the workload and uses the forced poweroff path.

## Guest service supervision

The measured root ships three CVM units: `cvm_bootstrap.service`,
`cvm_integrity.service` and `cvm_app.service`. Bootstrap performs the firewall,
clock, vault, sidecar and optional NFS gates, then notifies readiness before
starting the application units. Its supervisor runs fresh re-attestation children
every five minutes with a 300-second deadline. Tick status is recorded in
`/run/cvm/periodic.json`; acceptance tooling can request an immediate check with
`systemctl kill --kill-whom=main --signal=SIGUSR1 cvm_bootstrap.service`.

The distro `nftables.service` loads measured bootstrap rules before networking.
Docker socket activation is masked; provisioning configures the Docker daemon to
open its Unix socket directly. The independent integrity monitor retains its
watchdog. A security failure, or exit of either supervisor, makes PID 1 force
poweroff without a Python shutdown handler. This skips application graceful-stop
hooks on security failure. Development images use the same unit files and omit
TEE/KBS and integrity-monitor work.

Rebuild and reapprove generic CVM bundles after this change. Earlier hardware
acceptance records do not cover the new boot and shutdown sequence.

## Python package layout

The standalone `cvm` namespace has no NVFlare imports. Its packages separate
construction, guest execution, host launch, Trustee administration and artifact
transport:

| Package | Responsibility |
|---|---|
| `cvm.build` | Generic CVM construction, application vault sealing, build inputs and construction provisioning |
| `cvm.runtime` | Guest bootstrap, attestation, application lifecycle, GPU readiness, integrity and audit |
| `cvm.host` | Host capability checks, QEMU launch/shutdown and GPU assignment |
| `cvm.trustee` | Client/admin tools for the existing upstream CoCo Trustee; no server implementation |
| `cvm.artifacts` | Bundle/approval verification, OCI packaging and registry transport |
| `cvm.common` | Shared binding formats, measurement parsing, validation, pure policies and Linux primitives |

Guest images receive only `cvm.runtime` and the shared modules listed in
`cvm/build/payload.py`. Deliveries receive `cvm.host`, bundle verification and
their shared dependencies. Neither includes build or Trustee administration
code. The construction provisioner runs separately and is not installed in the
finished guest. Public shell commands and YAML fields are unchanged.

The package migration changes installed guest bytes and launcher templates.
Rebuild generic bundles with a new profile version, collect fresh measurements
and obtain new approval before creating deliveries with the reorganized tools.
Existing deliveries remain self-contained. The source fingerprint still covers
all construction sources, provisioning logic, payload lists and launch assets.

## Development and validation

Use a separate `dev-` profile plus `--dev` on both builders for an unencrypted
development vault and plain VM. Do not provide Trustee administration credentials to a dev
vault: `--dev` skips project discovery and rejects `--project-config`.
Dev artifacts cannot receive production approval or be mixed with production
deliveries. A test build requiring real TEE/KBS behavior instead uses a `test-`
profile and the explicit `--candidate` flag on Stage 2 and bundle administration.

Run the builder contracts on Linux with Python unittest. NVFlare's regular unit
suite also runs these contracts through
[`cvm_builder_test.py`](../../../../tests/unit_test/lighter/cvm_builder_test.py).
Use an isolated Linux test host and upstream Trustee for storage, HTTPS and hardware
acceptance; these tests are opt-in. Run these commands from the repository root:

```sh
export PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder:$PWD/tests/unit_test/lighter/cc/image_builder${PYTHONPATH:+:$PYTHONPATH}"
cargo build --locked --release --manifest-path tests/unit_test/lighter/cc/image_builder/policy_engine/Cargo.toml
python3 -m unittest discover -s tests/unit_test/lighter/cc/image_builder -v
sudo env PYTHONPATH="$PYTHONPATH" CVM_STORAGE_TESTS=1 \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_storage.py -v
```

`tests/integration_test/lighter/cc/image_builder/prepare_lab.py` and `tests/integration_test/lighter/cc/image_builder/lab_kbs.py` create an isolated test deployment
with disposable PKI and explicit loopback ports. Its HTTPS tests require
`CVM_HTTP_TESTS=1` and exercise actual Trustee encryption and policy evaluation
using signed fixtures. Such fixtures are protocol tests, not hardware acceptance.
Hardware acceptance must additionally exercise the exact sealed guest on each
platform, including corruption, interruption, binding, revocation and recovery.

Recorded versions, test results, artifact hashes and remaining production gates
are in [VALIDATION.md](VALIDATION.md). The checked-in profile is a template, not
an already approved production configuration.
