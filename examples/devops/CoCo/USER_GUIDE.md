# Bare-metal Confidential Containers administrator guide

## 1. Purpose and scope

This guide documents the staged `examples/devops/CoCo` scripts that build a
single-node Kubernetes cluster on an Ubuntu bare-metal server, install Kata
Confidential Containers with NVIDIA GPU passthrough, deploy local image trust
services, sign a workload image, and validate CPU, GPU, image, and encrypted
scratch-volume evidence from inside a confidential VM.

The supported confidential-computing paths are:

- AMD SEV-SNP with `kata-qemu-nvidia-gpu-snp`.
- Intel TDX with `kata-qemu-nvidia-gpu-tdx` and local PCCS/QGS services.

The scripts are a reproducible reference and validation environment. They are
not a production PKI, multi-node Kubernetes installer, or general-purpose
cluster lifecycle manager. Use a dedicated host. A production deployment must
move trust services and reference-value policy outside the workload host's
failure domain and apply the organization's controls for TLS, identity,
secrets, audit, backup, and change management.

Commands in this guide assume the working directory is the directory containing
`00-verify-host.sh`, `config.env.example`, `lib/`, and `templates/`.

This guide reaches a choice after shared Stage 30. Administrators can continue
here with the generic signed-workload attestation path, or switch to the
separate [`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md) guide for the alternative
NVFlare Stage 40/50 path.

## 2. Administrative ownership

The workflow crosses the host and Kubernetes administrative boundaries.

| Role | Responsibilities |
|---|---|
| Host IT administrator | Firmware configuration, Ubuntu lifecycle, IOMMU and KVM/TEE availability, storage health, network egress, package installation, systemd, kernel modules, swap, PCI/VFIO ownership, PCCS/QGS on TDX, reboot and recovery. |
| Kubernetes cluster administrator | kubeadm configuration, node readiness, CNI, RuntimeClass, Helm releases, node labels, GPU Operator, namespaces, Secrets, registry and Trustee deployments, workload policy, pod scheduling, and recovery review. |
| Security/attestation owner | Cosign key custody, registry CA, Intel PCS credentials if used, approved SNP/TDX reference values, Rekor policy, Trustee/RVPS policy, evidence retention, and acceptance criteria. |

One operator may hold all three roles in a lab. In a controlled environment,
separate them and require review of `config.env`, reference values, generated
keys, and host changes.

## 3. What the workflow builds

The normal control flow is:

```text
host preflight
    -> containerd + kubeadm + Calico
    -> Kata + Nydus + GPU Operator + VFIO
    -> private TLS registry + Cosign key + Trustee KBS
    -> copy and sign an immutable image digest
    -> launch a Kata confidential VM
         -> enforce measured image policy
         -> create encrypted confidential emptyDir
         -> collect and verify CPU evidence
         -> collect and verify NVIDIA GPU evidence
    -> write reports and delete the successful test pod
```

Pinned defaults are defined in `config.env.example`. At the time this suite was
validated, they included Kubernetes 1.34.9, containerd 2.2.2, Calico 3.32.1,
Kata Containers 3.29.0, NVIDIA GPU Operator 26.3.1, Cosign 2.6.2, NVIDIA NVAT
1.2.2, and a pinned Trustee KBS image.

Important host-side endpoints and paths are:

| Item | Default |
|---|---|
| Kubernetes API | `${NODE_IP}:6443` |
| Private registry | `${NODE_IP}:5000`, implemented as a pod `hostPort` |
| Trustee KBS | ClusterIP port 8080; lab configuration uses HTTP |
| Intel PCCS | Local HTTPS endpoint `127.0.0.1:8081` |
| Kubernetes admin configuration | `/etc/kubernetes/admin.conf` |
| Registry data | `/var/lib/coco-registry` |
| Suite-generated state | `state/` below the script directory |
| containerd binary | `/usr/local/bin/containerd` |
| containerd configuration | `/etc/containerd/config.toml` |
| Kata installation | `/opt/kata` |

## 4. Host prerequisites and change window

Before running any installation stage, confirm the following outside the
scripts:

1. The host is x86-64 Ubuntu and has a writable, healthy root filesystem.
2. Firmware enables the platform TEE:
   - AMD: SVM, SEV, SEV-ES/SEV-SNP, and the IOMMU.
   - Intel: TME, TME-MT, TDX, and VT-d/IOMMU.
3. The kernel exposes the matching KVM capability and `/dev/sev` or `/dev/kvm`.
4. At least one supported NVIDIA confidential-computing GPU is present. The
   host NVIDIA or Nouveau driver must not own it.
5. The GPU and any functions that must be assigned with it are in suitable
   IOMMU groups.
6. The host has at least 64 GiB RAM and 50 GiB free disk as a practical minimum.
7. Required HTTPS egress is permitted to Kubernetes, GitHub, GHCR, NVIDIA NGC,
   NVCR, Quay, Docker Hub, Sigstore when Rekor is enabled, AMD KDS for SNP, and
   Intel/Launchpad/keyserver endpoints for TDX.
8. The selected `NODE_IP` is stable and reachable from the host and Kata guest.
   It becomes the Kubernetes advertise address and a registry certificate SAN.
9. The operator is root or has passwordless, non-interactive `sudo`. Stages
   fail rather than pause for a sudo password.
Plan for a reboot or power cycle if the NVIDIA CC mode transition requires it.
Do not run this workflow on a host whose existing Kubernetes, containerd, CNI,
Kata, VFIO, or registry state must be preserved without first reviewing every
path modified by stages 10 through 50.

## 5. Configuration and precedence

Create the working configuration before the first run:

```bash
cp config.env.example config.env
chmod 0600 config.env
editor config.env
```

If `config.env` does not exist, `load_config` creates it automatically from the
example. Explicit review is preferable. Use `COCO_CONFIG=/absolute/path/file`
to select a different configuration file. `COCO_STATE_DIR` relocates generated
state for installation stages.

The common loader preserves environment overrides for `NODE_IP`, platform and
runtime selection, checksum/downgrade/tlog controls, attestation references,
Intel PCS key inputs, confidential-volume size, and pod retention. Other values
should be changed in `config.env` or through the documented positional arguments
to stage 40.

### 5.1 Platform and network settings

| Variable | Meaning and operational impact |
|---|---|
| `TEE_PLATFORM` | `auto`, `snp`, or `tdx`. `auto` checks CPU vendor and the `sev_snp` or `tdx_host_platform` CPU flag. Do not force a platform that the host does not provide. |
| `RUNTIME_CLASS` | `auto` selects the platform-specific NVIDIA Kata RuntimeClass. Override only for a reviewed Kata deployment. |
| `NODE_IP` | Kubernetes advertise address and default registry address. Empty means derive the source address for the default IPv4 route, then fall back to the first `hostname -I` address. |
| `POD_CIDR`, `SERVICE_CIDR`, `CLUSTER_DNS` | kubeadm and Calico network ranges. Check them against site routing, VPN, and management networks. |
| `KUBECONFIG_PATH` | Administrative kubeconfig used by all cluster operations. Presence of a nonempty file makes stage 10 skip `kubeadm init`. |

### 5.2 Version and supply-chain settings

`KUBERNETES_*`, `CONTAINERD_VERSION`, `CNI_PLUGINS_VERSION`, `HELM_VERSION`,
`CALICO_VERSION`, `KATA_VERSION`, `GPU_OPERATOR_VERSION`, `COSIGN_VERSION`,
`REGISTRY_IMAGE`, and `TRUSTEE_IMAGE` pin the deployed components. Stage 10
checks `CALICO_SHA256` before applying the cluster-admin manifest. Stage 30
requires immutable SHA-256 image references. The values are documented in
`CHECKSUMS.md`.

The three Kubernetes values serve different consumers:

- `KUBERNETES_MINOR` selects the Kubernetes APT repository, for example
  `v1.34`.
- `KUBERNETES_VERSION` is the exact Debian package version for kubelet,
  kubeadm, and kubectl, for example `1.34.9-1.1`.
- `KUBERNETES_SEMVER` is the upstream semantic version written into the kubeadm
  configuration, for example `v1.34.9`.

`CONTAINERD_SHA256`, `HELM_SHA256`, and `COSIGN_SHA256` authenticate the exact
download objects named in their comments. Do not substitute the hash of an
extracted executable for the hash of a compressed release archive.

`IGNORE_CHECKSUM_MISMATCH=1` permits a known checksum mismatch and is an unsafe
recovery control. It weakens supply-chain verification and must not be a normal
installation setting. `ALLOW_PACKAGE_DOWNGRADES=1` allows the pinned Kubernetes
packages to replace newer installed versions. Review the downgrade before using
it.

### 5.3 Registry, signing, and workload settings

| Variable | Meaning |
|---|---|
| `REGISTRY_PORT`, `REGISTRY_BIND_ADDRESS`, `REGISTRY_NAMESPACE`, `REGISTRY_DATA_DIR` | Host port, one specific host IPv4 address (defaults to `NODE_IP`), namespace, and host-backed data directory for the private TLS registry. |
| `REGISTRY_IMAGE`, `TRUSTEE_IMAGE` | Immutable digest-pinned infrastructure images; tag-only overrides are rejected. |
| `SECURITY_NAMESPACE`, `KBS_SERVICE` | Namespace and Service/Deployment name for Trustee KBS. |
| `SOURCE_IMAGE` | Image copied into the local registry by stage 40. |
| `TARGET_IMAGE_NAME` | Repository path below `${REGISTRY_HOST}`. |
| `COSIGN_TLOG_MODE` | `disabled` verifies only the configured public key; `rekor` uploads and requires public transparency-log evidence. Rekor reveals signature metadata and requires Sigstore connectivity. |
| `ROTATE_SECURITY_MATERIAL` | `1` backs up and replaces the Cosign pair and registry CA/certificates during stage 30. Return it to `0` after intentional rotation. |
| `GPU_RESOURCE`, `GPU_COUNT`, `POD_MEMORY` | Resource request used by the attestation pod. The default GPU resource is `nvidia.com/pgpu`. |
| `CONFIDENTIAL_VOLUME_SIZE` | Positive binary Kubernetes quantity for the encrypted confidential `emptyDir`. Allow headroom for ext4 and integrity metadata. |
| `KEEP_ATTESTATION_POD` | `0` deletes a successful validation pod; `1` leaves it running for inspection and keeps its passthrough GPU allocated. |

### 5.4 CPU attestation reference values

`EXPECTED_SNP_LAUNCH_MEASUREMENT` must be 96 lowercase hexadecimal characters
representing the approved 48-byte SNP launch measurement.
`AMD_KDS_PRODUCT` selects the AMD Key Distribution Service product path used to
retrieve the matching VCEK and certificate chain; the default is `Genoa` and
must match the processor generation represented by the report.

TDX requires all five 48-byte references:

- `EXPECTED_TDX_MRTD`
- `EXPECTED_TDX_RTMR0`
- `EXPECTED_TDX_RTMR1`
- `EXPECTED_TDX_RTMR2`
- `EXPECTED_TDX_RTMR3`

Obtain these values from an independently trusted build manifest, release
process, or Trustee/RVPS appraisal policy. A measurement observed from the same
untrusted host during the current run is diagnostic information, not an
approved reference.

Complete VM-image verification requires comparing the launch measurements with
trusted references, normally through Trustee/RVPS policy, or enforcing an SNP
launch with a signed ID block. The local verifier in this suite performs direct
comparison with protected configuration values; production policy should live
outside the workload host's failure domain.

### 5.5 Intel TDX quote-service settings

`TDX_ATTESTATION_PPA_SUITE` chooses the Canonical TDX attestation PPA suite.
Empty means the host codename. An older-suite override is an explicit
compatibility decision and should be tested as part of OS qualification.

Intel production PCS can operate without a subscription key in the validated
topology. If the PCS environment requires a key, prefer
`INTEL_PCS_API_KEY_FILE` pointing to a root-readable file containing one
32-character alphanumeric key. `INTEL_PCS_API_KEY` is accepted as a one-run
environment value. Set only one. Never commit the key, include it in an image,
or print it in logs.

## 6. Recommended execution procedure

Run the stages individually during initial qualification so that change records
and failure boundaries are clear:

```bash
./00-verify-host.sh
./10-install-kubernetes.sh
./20-install-coco-gpu.sh
./30-deploy-security-services.sh
```

At this point choose one Stage 40/50 direction. For the generic attestation
validation described by this guide, continue with:

```bash
./40-sign-image.sh
./50-launch-and-attest.sh
```

For an NVFlare server and GPU client, use
[`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md) instead. Do not run both Stage 40/50
variants or select them with a wildcard.

After the host has been qualified and the configuration placed under change
control, `./run-all.sh` runs stages 00 through 50 in that order. All scripts use
`set -Eeuo pipefail` and stop on the first unhandled error.

## 7. Script reference

### 7.1 `run-all.sh`

This is the sequential orchestrator. It resolves its own directory, prints a
banner for each stage, and invokes stages 00, 10, 20, 30, 40, and 50. It does
not run the alternative NVFlare Stage 40/50/60 path.

There is no rollback transaction. If a stage fails, correct the cause and rerun
that stage, then continue. Earlier stages are designed to be mostly restartable,
but they still rewrite suite-owned configuration and may restart services.

Success is the message `End-to-end CoCo deployment and attestation completed`
and a path to `state/latest-attestation-reports`.

### 7.2 `00-verify-host.sh`

Purpose: perform a mostly read-only host readiness assessment before package,
cluster, or GPU changes.

Detailed behavior:

1. Loads configuration, detects SNP or TDX, derives `NODE_IP`, and selects the
   platform RuntimeClass.
2. Creates and deletes a temporary file under `/etc` through `sudo` to prove the
   root filesystem accepts writes.
3. If root is ext4 and the kernel exports `errors_count`, fails when recorded
   filesystem errors are nonzero.
4. Checks x86-64 and reports the Ubuntu version.
5. For SNP, checks AMD vendor, `sev_snp`, KVM AMD SEV enablement, and `/dev/sev`.
6. For TDX, checks Intel vendor, `tdx_host_platform`, KVM Intel TDX enablement,
   and `/dev/kvm`.
7. Requires populated IOMMU groups.
8. Finds the first NVIDIA VGA/3D controller, recognizes H100/H200 and selected
   Blackwell families, checks driver ownership and IOMMU isolation, and rejects
   loaded host NVIDIA/Nouveau modules or installed `nvidia-driver-*` packages.
9. Reports swap, memory, free disk, and all required network endpoints.

An unbound GPU is valid before stage 20. A GPU already bound to `vfio-pci` is
also valid. `FAIL` increments the failure count and makes the script exit
nonzero; `WARN` is advisory. The expected gate is:

```text
Summary: 0 failure(s), 0 warning(s)
```

The script may create `config.env` and `state/` through the shared loader and
performs the transient `/etc` write probe; it does not install software.

### 7.3 `10-install-kubernetes.sh`

Purpose: establish the single-node Kubernetes and container-runtime baseline.

Host and package actions:

1. Rejects an implicit downgrade when installed kubelet, kubeadm, or kubectl is
   newer than the pinned version.
2. Installs administrative dependencies. TDX also installs Go when absent;
   native `/usr/bin/runc` is installed when absent.
3. Loads `overlay` and `br_netfilter`, writes `/etc/modules-load.d/k8s.conf`,
   writes Kubernetes forwarding/bridge sysctls, disables swap, and comments
   active swap entries in `/etc/fstab` with a suite marker.
4. Downloads the pinned containerd release archive into `state/downloads`,
   verifies its SHA-256, and extracts it below `/usr/local`.
5. Downloads CNI plugins and their publisher checksum, verifies them, and
   extracts them into `/opt/cni/bin`.

containerd configuration:

- Writes version-3 configuration to `/etc/containerd/config.toml`.
- Uses `/var/lib/containerd` and `/run/containerd`.
- Imports suite fragments from `/etc/containerd/conf.d/*.toml` and Kata
  fragments from `/opt/kata/containerd/config.d/*.toml`.
- Enables CDI and configures the registry-certificate directory.
- Configures systemd cgroups and explicitly pins the OCI runtime to the trusted
  `/usr/bin/runc`, avoiding an unmanaged binary earlier in `PATH`.
- Installs `/etc/systemd/system/containerd.service` with
  `/usr/local/bin/containerd`, enables it, restarts it, and requires it active.

Kubernetes actions:

1. Installs the Kubernetes APT key and repository.
2. Installs exact kubelet, kubeadm, and kubectl versions, holds them, and
   explicitly enables kubelet so the cluster survives reboot.
3. Downloads and checksum-verifies Helm into `/usr/local/bin/helm`.
4. If `KUBECONFIG_PATH` is missing or empty, renders `templates/kubeadm.yaml.in`
   and runs `kubeadm init`. The template uses systemd cgroups, the selected node
   address and CIDRs, a 20-minute runtime request timeout, and the
   `KubeletPodResourcesGet` and `RuntimeClassInImageCriApi` feature gates.
5. Downloads and applies the pinned Calico manifest, removes the control-plane
   scheduling taint for the single node, and waits for Ready.
6. Copies the admin kubeconfig to the invoking non-root user's `~/.kube/config`.
On rerun, containerd configuration is rewritten and containerd is restarted.
An existing nonempty kubeconfig suppresses `kubeadm init`; it is therefore the
operator's responsibility to ensure that file describes the intended cluster.
The success marker is `Kubernetes is ready`.

### 7.4 `20-install-coco-gpu.sh`

Purpose: add platform attestation support, Kata/Nydus, NVIDIA GPU Operator, and
the platform-specific confidential GPU RuntimeClass.

The script requires a working Kubernetes API, Helm, containerd/ctr, and PCI
inventory. It selects the first Kubernetes node and applies the platform TEE
label plus `nvidia.com/gpu.workload.config=vm-passthrough`.

#### Intel TDX PCCS/QGS path

On TDX only, the script:

1. Resolves the host or explicitly configured PPA suite and verifies that the
   PPA publishes it.
2. Downloads the Canonical PPA signing key and requires fingerprint
   `0C0E6AF955CE463C03FC51574D098D70AFBE5E1F` before installing it.
3. Installs PCCS, QGS, DCAP QPL, RA service, and PCK-ID tools.
4. Validates the optional PCS key. A configured key is installed as
   `root:pccs` mode `0640` and is checked for readability by the `pccs` account.
5. In keyless mode, applies a guarded PCCS 1.21 JavaScript compatibility change
   that removes an empty subscription header. The edit requires an exact source
   anchor and must pass `node --check`.
6. Enables and restarts `pccs` and `qgsd`, waits for PCCS HTTPS on loopback, and
   requires QGS active.

#### Kata, Nydus, and encrypted storage capability

The script installs the pinned Kata OCI Helm chart, then waits for the Kata shim
and the platform runtime fragment on the host. It repairs the suite-owned
containerd import if an older stage-10 run omitted it, restarts containerd, and
restarts kubelet to clear Kubernetes 1.34's cached CRI cgroup-driver response.
It then requires:

- the platform RuntimeClass in the effective containerd configuration;
- the `nydus-for-kata-tee` snapshotter plugin in `ok` state; and
- `emptydir_mode = "block-encrypted"` in the resolved Kata runtime
  configuration.

That setting enables the released confidential `emptyDir` implementation used
in stage 50. It does not create a persistent encrypted volume.

#### NVIDIA GPU Operator path

The GPU Operator Helm release enables sandbox workloads in Kata mode, Node
Feature Discovery, CC Manager with default mode `on`, VFIO Manager, and the Kata
sandbox device plugin. Kata Manager is disabled because the upstream Kata chart
already installed Kata.

The script waits up to 30 minutes for the CC-ready node label and allocatable
`nvidia.com/pgpu` resource. It enumerates every NVIDIA VGA/3D PCI device and
fails unless every one is bound to `vfio-pci`.

Some GPU CC-mode changes require a reboot. If the operator reports that state,
reboot, rerun stage 20, and wait for `Kata CoCo runtime and GPU passthrough are
ready` before proceeding.

### 7.5 `30-deploy-security-services.sh`

Purpose: establish the local registry, signing material, image-verification
policy, and Trustee resource service used by the validation workflow.

Cosign and key handling:

- Installs the pinned Cosign binary after SHA-256 verification.
- Stores sensitive material under `state/security` and `state/registry-tls`,
  both mode `0700`.
- Generates `cosign.key` mode `0600` and `cosign.pub` mode `0644` when absent.
- Uses `COSIGN_PASSWORD` if a password-protected signing key is desired.
- With `ROTATE_SECURITY_MATERIAL=1`, moves existing keys and certificates to
  UTC-stamped backups before generating replacements.

Registry PKI behavior:

- Creates a 4096-bit RSA private CA valid for 3650 days when the CA is missing,
  expiring, mismatched, or invalid.
- Issues a 3072-bit RSA leaf certificate valid for 825 days with an IP or DNS
  SAN matching `REGISTRY_HOST`.
- On rerun, verifies expiry, key match, CA signature, and SAN. A bad leaf is
  replaced without rotating a healthy CA; replaced files are timestamped.

Kubernetes and host behavior:

1. Creates the registry namespace and TLS Secret.
2. Renders `templates/registry.yaml.in`. The digest-pinned registry uses a
   hostPath data directory, binds `hostPort` only on
   `REGISTRY_BIND_ADDRESS`, and uses a Recreate strategy. It deliberately has
   no client authentication and permits push/delete; restrict that address and
   port to trusted administrators with host/network firewall policy.
3. Installs the registry CA and `hosts.toml` below
   `/etc/containerd/certs.d/${REGISTRY_HOST}` and restarts containerd.
4. Builds a containers/image policy whose default action is reject and whose
   only allow rule is `sigstoreSigned` for `TARGET_REPOSITORY` with repository
   identity matching.
5. Stores the policy, Cosign public key, and registry CA in a Kubernetes Secret.
6. Deploys digest-pinned Trustee KBS with built-in attestation/RVPS support and
   read-only verification resources. The resource policy is mounted directly
   from its ConfigMap; no mutable-tag init image can rewrite KBS state.
7. Optionally stages NVFlare hardware-authorizer credentials in
   `NVFLARE_NAMESPACE`. Before writing either credential Secret, it creates the
   dedicated namespace with `nvflare.nvidia.com/coco-managed=true` or requires
   that exact label on an existing namespace. It refuses to adopt an unlabeled,
   potentially shared namespace. On TDX,
   `NVFLARE_TDX_ATTESTATION_CONFIG_FILE` must name a readable JSON object with
   nonempty `trustauthority_url`, HTTPS `trustauthority_api_url`, and nonempty
   `trustauthority_api_key` fields; the script creates the
   `nvflare-tdx-attestation` Secret without printing the file. An optional
   `NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE` must contain one nonempty line of
   at most 4096 characters and becomes the `nvflare-gpu-attestation` Secret.

The TDX Trust Authority attestation API key above is not an Intel PCS
subscription key. If the TDX configuration is omitted, this generic workflow
remains usable, but the alternative NVFlare Stage 50 fails closed. Rerun Stage
30 after setting the file to stage the Secret. NVFlare Stage 50 defaults to
local NVAT verification and needs no NVIDIA service key. Remote NRAS mode is an
explicit opt-in and may require the staged NVIDIA service key.

The sample KBS uses HTTP, an insecure token key, ephemeral `emptyDir` state,
and a deny-all admin interface. This is suitable only for the local validation
topology. Stage 50 embeds the public policy, public key, and registry CA directly
in measured init-data for early guest image-pull enforcement; the KBS endpoint
is available for later resource workflows but is not required for that public
material.

The success marker is `Registry and Trustee KBS are ready`.

#### Choose the Stage 40/50 direction

Stop after this success marker and choose one of these mutually exclusive
continuations:

1. **Generic attestation validation:** continue in this document with
   `40-sign-image.sh` and `50-launch-and-attest.sh`. This path signs one
   configured workload image, launches a GPU-enabled confidential VM, and
   produces CPU, GPU, image-integrity, and encrypted-storage attestation
   reports.
2. **NVFlare deployment:** switch to
   [`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md). That guide builds and signs
   separate NVFlare server and client images, then deploys a CPU-only server
   and a one-GPU client as confidential VMs. It assumes shared stages 00–30 are
   already complete and covers only its alternative Stages 40 through 60,
   including an end-to-end `hello-numpy` job check.
   Stages 30 and 50 create the dedicated `NVFLARE_NAMESPACE` with
   `nvflare.nvidia.com/coco-managed=true` and refuse an existing namespace that
   lacks that ownership label. This fail-closed check occurs before credential
   Secrets are written or Pod Security Admission is changed. Never add the
   ownership label to a shared namespace. Stage 50 then labels the owned
   namespace with
   `pod-security.kubernetes.io/enforce=privileged` before deployment because
   both Kata Pods require `privileged: true` to create the guest-local TEE
   device. Restrict Pod-creation RBAC in that namespace; the admission label
   applies to every Pod there, not only to Kata-isolated workloads.

   Stage 50 runs the following operation automatically (with the configured
   kubeconfig and namespace):

   ```bash
   kubectl label namespace "$NVFLARE_NAMESPACE" \
     pod-security.kubernetes.io/enforce=privileged --overwrite
   ```

Do not run both continuations or use a wildcard such as `40-*.sh` or
`50-*.sh`; the scripts at each number are alternative workload paths.

### 7.6 `40-sign-image.sh`

Purpose: copy a source image into the local TLS registry, sign its immutable
digest, verify the signature, and publish state for stage 50.

Invocation forms are:

```bash
./40-sign-image.sh
./40-sign-image.sh SOURCE_IMAGE TARGET_TAG
```

The default source is `SOURCE_IMAGE` and the default target tag is `signed`.
The script:

1. Requires the stage-30 Cosign private key and registry CA.
2. Copies the CA to a containers-certs directory used by Skopeo.
3. Uses Skopeo to copy the image to `${TARGET_REPOSITORY}:${TARGET_TAG}`.
4. Resolves the registry digest and signs `${TARGET_REPOSITORY}@sha256:...`, not
   the mutable tag.
5. Uploads and verifies Rekor evidence when `COSIGN_TLOG_MODE=rekor`; otherwise
   signs without tlog upload and verifies with the configured public key while
   printing an explicit transparency notice.
6. Writes `state/cosign-verification.json`,
   `state/image-signing-report.txt`, and shell-quoted
   `state/signed-image.env` for stage 50.

Rerunning with the same tag can replace the tag target and add another
signature, but stage 50 consumes the last verified digest saved in
`signed-image.env`. Protect `cosign.key`; do not print, commit, or embed it in a
container image. The success marker begins `Signed image ready:`.

### 7.7 `50-launch-and-attest.sh`

Purpose: launch the signed workload in the selected Kata confidential VM and
produce fail-closed CPU, GPU, image, and encrypted-storage verification reports.

Before launching, the wrapper requires:

- `state/signed-image.env` from stage 40;
- correctly formatted platform reference values;
- a valid confidential-volume size;
- the platform RuntimeClass in effective containerd configuration; and
- the Nydus proxy snapshotter in `ok` state.

The optional first argument is the report directory. Otherwise a UTC-stamped
directory is created below `state/`. The wrapper passes configuration to
`lib/deploy-coco-attest.sh`, restores report ownership to the invoking user, and
updates `state/latest-attestation-reports`.

#### Internal pod and image-policy flow

The driver first verifies the digest-pinned image from the host. It creates
`initdata.toml` containing:

- an offline filesystem KBC configuration;
- the deny-by-default `sigstoreSigned` image policy and Cosign public key;
- the private registry CA and registry configuration; and
- a Kata agent policy allowing the operations required by this validation pod.

The measured demo agent policy permits host-admin copy, exec, and stream access
because the validation scripts use `kubectl exec` and read command output. It
denies `SetPolicyRequest`, so the host cannot replace that measured policy at
runtime. This is an operational demo boundary, not workload isolation from a
cluster administrator; production policy should remove exec/copy/stream rules
that the workload does not need.

The file is gzip/base64 encoded in the
`io.katacontainers.config.hypervisor.cc_init_data` annotation. Its uncompressed
SHA-256 is later compared with SNP `HOST_DATA` or the first 32 bytes of TDX
`MRCONFIGID`. The pod uses the selected RuntimeClass, requests the configured
GPU and memory, and is privileged inside its isolated Kata VM so it can use the
guest attestation interfaces. Privilege inside the VM is not equivalent to a
host-privileged runc container.

The pod reaches Ready only after guest `image-rs` accepts the signed image under
the measured deny-by-default policy.

#### Confidential scratch-volume verification

The pod receives a confidential `emptyDir`. Kata creates a host-backed sparse
device; inside the guest it generates an ephemeral key and applies LUKS2,
dm-crypt, dm-integrity, and ext4. The driver requires:

- a writable ext4 mount backed by `/dev/mapper/*`;
- a `CRYPT-LUKS2-*` device-mapper UUID;
- an underlying `INTEGRITY-*` mapping; and
- a successful random write, sync, and readback probe.

This volume is ephemeral and is removed with the pod. It is not an encrypted
persistent volume. Its key and detached header remain in TEE-protected guest
memory. dm-integrity detects block modification but does not prevent replay of
an older complete volume state.

#### AMD SEV-SNP CPU verification

The driver compiles a small collector on the host and copies it into the guest.
It requests the fixed 1,184-byte SNP report with a fresh 64-byte challenge. On
the host it:

1. checks `REPORT_DATA` against the challenge;
2. checks `HOST_DATA` against measured init-data SHA-256;
3. parses the signed launch measurement and TCB/chip values;
4. retrieves the matching AMD VCEK and certificate chain from AMD KDS;
5. verifies the VCEK chain and ECDSA P-384/SHA-384 report signature; and
6. compares the signed launch measurement with
   `EXPECTED_SNP_LAUNCH_MEASUREMENT`.

An authentic signature without a matching approved measurement fails.

#### Intel TDX CPU verification

The driver checksum-verifies source for a pinned `google/go-tdx-guest` commit,
builds its collector and verifier, and copies only the collector into the
guest. It obtains a fresh TD Quote through configfs/TSM and QGS, then runs the
host verifier with Intel collateral and CRL checking enabled. It requires:

1. quote signature and embedded certificate-chain verification;
2. acceptable Intel PCS collateral and TCB status;
3. the fresh 64-byte challenge in `REPORTDATA`;
4. measured init-data SHA-256 plus zero padding in `MRCONFIGID`;
5. exact MRTD and RTMR0-3 reference matches; and
6. the TD debug attribute disabled.

The raw parser supports TDX quote versions 4 and 5 and independently extracts
the claims used in the human-readable report.

#### NVIDIA GPU verification

The driver checksum-verifies the pinned NVIDIA NVAT repository package, copies
it into the guest, installs `nvattest`, and generates a fresh 32-byte GPU nonce.
It collects evidence and performs local NVAT appraisal. A pass requires, among
other claims:

- result code zero and message `Ok`;
- nonce match;
- successful measurement appraisal;
- secure boot enabled and debug disabled;
- report signature verified;
- valid attestation certificate and good OCSP status;
- driver and VBIOS RIM signature/version matches; and
- no mismatched measurement records.

#### Reports, cleanup, and failure behavior

The driver writes CPU, GPU, image, and storage reports plus raw evidence and a
`SHA256SUMS` manifest. With the default `KEEP_ATTESTATION_POD=0`, the wrapper
requests `--cleanup`, but the trap deletes the pod only after all verification
has succeeded. A failed pod is intentionally left for `kubectl describe`, logs,
and guest inspection. Delete it manually after collecting diagnostics so the
passthrough GPU can be allocated again.

Success requires all human-readable reports to contain an overall `PASS`.

### 7.8 `lib/deploy-coco-attest.sh`

This is the implementation driver used by stage 50. Cluster administrators may
invoke it directly for a different namespace, pod name, output directory, or
kubeconfig:

```bash
lib/deploy-coco-attest.sh \
  --namespace default \
  --pod-name coco-attestation-manual \
  --output-dir ./manual-reports \
  --kubeconfig /etc/kubernetes/admin.conf \
  --cleanup
```

Direct invocation requires all environment inputs that stage 50 normally
provides: digest-pinned image, policy repository, registry host and CA, Cosign
public key, RuntimeClass, platform, approved measurement references, resources,
and artifact cache. Prefer the wrapper unless debugging or integrating with an
external automation system.

### 7.9 `lib/common.sh`

This shared library is sourced by stages 00 through 50. It:

- resolves suite, configuration, and state paths;
- detects SNP versus TDX and selects labels/RuntimeClass;
- validates boolean and enum controls;
- derives node, registry, and repository addresses;
- implements root, kubectl, and Helm wrappers;
- downloads artifacts atomically through process-specific `.partial` files;
- verifies pinned SHA-256 values and rejects mismatches by default; and
- supplies bounded polling and passwordless-sudo checks.

It is not intended as a standalone command.

## 8. Generated state and evidence

Treat `state/` as sensitive administrative state:

| Path | Contents |
|---|---|
| `state/downloads/` | Persistent artifact cache and pinned TDX verifier source. |
| `state/security/cosign.key` | Image-signing private key; mode `0600`. |
| `state/security/cosign.pub` | Public verification key. |
| `state/registry-tls/` | Private CA and registry leaf key/certificate. |
| `state/signed-image.env` | Last verified immutable workload image reference. |
| `state/cosign-verification.json` | Host-side Cosign verification output. |
| `state/latest-attestation-reports` | Symlink to the latest report directory. |

The report directory includes:

- `cpu-attestation-report.bin` and `.txt`;
- `gpu-attestation-evidence.json`;
- `gpu-attestation-result.json`;
- `gpu-attestation-report.txt`;
- `gpu-confidential-compute-status.txt`;
- `image-verification-report.txt`;
- `storage-verification-report.txt`;
- `initdata.toml` and the applied pod manifest; and
- `SHA256SUMS` covering the material verification reports.

The checksums detect later report-file changes; they do not timestamp or
externally notarize evidence. Copy required reports to controlled storage and
apply the site's retention and signing policy.

## 9. Operational verification commands

After stage 10:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes -o wide
systemctl is-enabled containerd kubelet
systemctl is-active containerd kubelet
sudo containerd config dump | sed -n '/runtimes.runc.options/,/^[[:space:]]*\[/p'
```

After stage 20:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get runtimeclass
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get node -o json
sudo ctr plugins ls | grep nydus-for-kata-tee
lspci -Dnnk -d 10de:
```

On TDX:

```bash
systemctl is-active pccs qgsd
curl --insecure --fail https://127.0.0.1:8081/ >/dev/null
```

After stages 30 and 40:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get deployments -A
curl --fail --cacert state/registry-tls/ca.crt "https://${NODE_IP}:5000/v2/"
sed -n '1,120p' state/image-signing-report.txt
```

After stage 50:

```bash
cd state/latest-attestation-reports
sha256sum -c SHA256SUMS
grep -H 'Overall .* result: PASS' *-report.txt
```

## 10. Failure handling and reruns

- Preflight failure: correct firmware, kernel, storage, driver, IOMMU, DNS,
  proxy, or firewall state before installation.
- Kubernetes failure: inspect `systemctl status containerd kubelet`, their
  journals, `kubeadm` output, CNI state, and CIDR conflicts. Rerun stage 10 after
  correction; it does not reset an existing cluster.
- Kata/GPU failure: inspect Helm releases and pods in `kata-system` and
  `gpu-operator`, node labels/capacity, containerd's effective config, Nydus
  plugin state, and PCI driver ownership. Reboot if CC Manager requests it,
  then rerun stage 20.
- Registry/Trustee failure: validate the configured node address, registry SAN,
  host port availability, registry hostPath permissions, Kubernetes Secrets,
  and deployment logs. Rerun stage 30; healthy keys and CA are reused.
- Signing failure: validate registry TLS, source registry access, key password,
  image reference, and Rekor connectivity/policy. Rerun stage 40.
- Attestation failure: keep the failed pod, inspect `kubectl describe`, validate
  QGS/PCCS or AMD KDS access, check the approved references, and review raw
  reports. Never solve a reference mismatch by copying the observed value into
  trusted configuration without an independent approval process.

The workflow has no rollback transaction. Use a separately reviewed recovery
procedure, or reimage the dedicated host when an exact trusted baseline is
required.

## 11. Production hardening gaps

Before adapting the example for production, at minimum address:

- external, authenticated TLS for Trustee KBS and the registry;
- protected Cosign identity or KMS/HSM-backed signing;
- formal transparency and audit policy;
- external Trustee/RVPS reference values and restrictive appraisal policy;
- persistent, backed-up KBS state and controlled secret release;
- least-privilege Kubernetes RBAC and namespace/network policies;
- image and chart mirroring with organization-controlled provenance;
- patch and upgrade ownership for all pinned components;
- evidence export, timestamping, retention, and incident response; and
- a validated host reimage/recovery procedure.
