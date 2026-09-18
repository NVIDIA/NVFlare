# Run a CVM with an application vault

Each platform/application delivery is an OCI artifact containing its complete
reusable generic CVM bundle plus one application vault. The OCI manifest has a
reusable CVM layer and a deployment-specific vault/runtime layer. A local OCI
layout tar or its immutable registry digest is sufficient to launch; no generic
bundle needs to be downloaded separately. Materialization restores the familiar
platform directory with `cvm_bundle/`, and the launcher verifies that exact
combination.
A GPU profile selects the GPU, passes it to the VM, and exposes it to the
application container automatically.

The backend operator can follow [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md) to set up
Trustee, authorize the delivery's CVM bundle, and manage its vault key.

## 1. Prepare the runtime host

Use an SNP host for an `amd_sev_snp` delivery or a TDX host with an operational
quote-generation service for an `intel_tdx` delivery. Install Python 3 and QEMU.
The host needs KVM access, enough memory for the profile, and guest network access
to the configured KBS.

Obtain the platform's `.oci.tar` from Vault Build or its immutable registry
reference. Install [ORAS](https://oras.land/docs/installation/) only when using a
registry; local OCI tar materialization uses Python alone.

## 2. Materialize the delivery artifact

Use `scripts/cvm_pull` to validate every OCI descriptor and layer digest and
materialize a private directory. Its platform directory contains both manifests
and everything needed to launch:

```text
/srv/cvm/vault_my-app-site1/
└── intel_tdx/
    ├── cvm_bundle/
    │   ├── cvm_manifest.json
    │   ├── OVMF.fd
    │   ├── vmlinuz
    │   ├── initrd.img
    │   └── verity_root.qcow2
    ├── vault_manifest.json
    ├── vault.qcow2
    ├── launch_cvm.sh
    └── shutdown_cvm.sh
```

### From an OCI layout tar

Vault Build generates the archive's deployment ID automatically. Use the filename
provided by the builder; the ID below is illustrative. The `--output` folder is
an operator-selected local name and can remain readable.

```sh
sha256sum vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar
sudo scripts/cvm_pull vault_0123456789ab4def8123456789abcdef_intel_tdx.oci.tar \
  --output /srv/cvm/vault_my-app-site1
```

Compare the first digest with `archive_sha256` in the publisher's
`oci_artifacts.json` before materializing. The tar is a standard OCI image layout;
its payload layers are already compressed.

### From an OCI registry URL

Use the manifest digest printed by `scripts/cvm_publish`. A digest is required;
the pull wrapper refuses a mutable tag.

```sh
artifact=registry.example.org/cvm/my-app-site1@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
sudo scripts/cvm_pull "$artifact" --output /srv/cvm/vault_my-app-site1
```

For an isolated lab registry that deliberately uses HTTP, opt in explicitly:

```sh
sudo scripts/cvm_pull \
  registry.example.org:5000/cvm/my-app-site1@sha256:OCI_MANIFEST_DIGEST \
  --plain-http --output /srv/cvm/vault_my-app-site1
```

For production, verify the publisher's signature before materializing, using the
same immutable reference:

```sh
cosign verify --key /etc/cvm/release-signing.pub "$artifact"
```

OCI digest validation detects corruption and substitution within a selected
artifact. The signature or another authenticated release channel establishes
who selected that digest.

## 3. Launch CVM

Change to the delivery's platform directory and run:

```sh
cd /srv/cvm/vault_my-app-site1/intel_tdx
sudo ./launch_cvm.sh
```

The launcher reads the included `cvm_bundle/` whose manifest has the required
`cvm_build_id`, checks the local TEE, verifies every generic artifact, checks the
vault binding, locks the actual vault inode, records runtime state, and starts
QEMU.

For a GPU profile, it finds exactly the profile's `gpu_count` NVIDIA display controllers with isolated IOMMU groups, loads `vfio-pci`, binds every function in each GPU slot, and passes all selected slots to the VM. TDX uses one QEMU IOMMUFD backend for these assignments. The guest application
is started by Docker with `--gpus all`. The
launcher restores the original host drivers when QEMU exits. For explicit placement, repeat `--gpu` exactly `gpu_count` times:

```sh
sudo ./launch_cvm.sh --gpu 0000:41:00.0 --gpu 0000:43:00.0
```

Advanced operators may override the embedded bundle with a separately managed
shared bundle directory:

```sh
sudo ./launch_cvm.sh \
  --cvm-bundle /opt/cvm-bundles/cvm_cpu-2026.09/intel_tdx
```

Every selected GPU must be isolated from other PCI slots in its IOMMU group. The
launcher refuses to detach a mixed group because doing so could interrupt an
unrelated host device.

During boot, the guest verifies the dm-verity root, resolves disks by fixed SCSI
role, checks the complete vault-header binding against local TEE evidence,
obtains the secret after KBS appraisal, opens and scans the authenticated vault,
then mounts clear `/applog` read/write and clear `/user_config` and `/user_data`
read-only before starting the container.

Writes under `/` use the RAM-backed overlay and count against
`root_overlay_max_mib`. `/vault`, `/applog`, `/user_config`, `/user_data`, and the
separate `/tmp` tmpfs do not consume that limit. Use `df -h /cow` in the guest to
inspect it.

## 4. Connect to the application

Only configured ports are forwarded and admitted by the guest firewall. For the
sample vault:

```sh
curl http://127.0.0.1:8080/
```

The application receives `/vault`, `/applog`, `/user_config`, `/user_data`, and
`/host/bin`. Only `/vault` is encrypted and authenticated at rest. `/applog` is
the CVM's clear output-only channel, intended for logs the operator must read
without a key. `/user_config` and `/user_data` are clear operator inputs enforced
read-only by QEMU, the guest mount, and the container bind mount. Treat all
sidecar content as public and untrusted; keep secrets and confidential logs in
`/vault`.

## 5. Shutdown CVM

From another shell in the same delivery directory, run:

```sh
sudo ./shutdown_cvm.sh
```

The script validates the root-owned runtime record, signals the exact launcher,
waits for QEMU to exit, and leaves the launcher holding the vault lock until that
process has stopped. `Ctrl-C` in the launch terminal remains supported.

Wait for shutdown to finish before copying, moving, backing up, or inspecting any
writable disk. If no matching CVM is running, the shutdown command fails without
signalling another process.

## 6. Keep vault attachment exclusive

One `vault.qcow2` file may be attached to only one CVM at a time. A second launch
of the same inode is refused. You may copy a delivery after its CVM is fully
stopped. Each copy can then run independently, but every byte-for-byte copy keeps
the same key authorization and revocation scope.

A failed root check, vault binding, appraisal, key retrieval, authenticated scan,
integrity monitor, periodic appraisal, or key revocation stops the workload and
powers off the guest. After shutdown, an operator can mount `applog.qcow2`
directly and read the public logs without a KBS key.

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for creation and
[VALIDATION.md](VALIDATION.md) for acceptance testing.


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
