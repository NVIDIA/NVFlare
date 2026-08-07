# Bare-metal CoCo scripts: quick trial summary

This is the short path for trying the scripts on a dedicated Ubuntu bare-metal
host with AMD SEV-SNP or Intel TDX and a supported NVIDIA confidential-computing
GPU. For exact host changes, security assumptions, and troubleshooting, read
[`USER_GUIDE.md`](USER_GUIDE.md).

To deploy an NVFlare server and client after preparing the CoCo cluster, use
the alternate Stage 40/50 procedure in [`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md).

## Before you start

These scripts install and configure containerd, Kubernetes, Calico, Kata,
NVIDIA GPU Operator, a private registry, and Trustee. They disable swap, change
host networking/sysctls, bind the GPU to `vfio-pci`, and create signing keys.
Use a dedicated or disposable host and an account with passwordless `sudo`.

Firmware must already enable:

- AMD SVM/SEV/SNP and IOMMU, or Intel TME/TME-MT/TDX and IOMMU.
- NVIDIA GPU confidential-computing support.

The host must not have an active NVIDIA or Nouveau driver owning the GPU.

## Configure

From the directory containing the scripts:

```bash
cp config.env.example config.env
chmod 0600 config.env
editor config.env
```

Check these values first:

- `NODE_IP`: leave empty only if the default-route address is stable.
- `TEE_PLATFORM=auto`: normally detects SNP or TDX.
- `SOURCE_IMAGE`: image to copy and sign.
- `COSIGN_TLOG_MODE`: `disabled` for the private lab flow or `rekor` for public
  transparency-log upload and verification.
- `EXPECTED_SNP_LAUNCH_MEASUREMENT`: independently approved SNP measurement.
- `EXPECTED_TDX_MRTD` and `EXPECTED_TDX_RTMR0` through `RTMR3`: independently
  approved TDX references; stage 50 refuses empty values.
- `TDX_ATTESTATION_PPA_SUITE`: set only when an explicitly approved compatible
  PPA suite is required.
- `INTEL_PCS_API_KEY_FILE`: usually empty for production-PCS keyless mode. If a
  key is required, point to a root-readable file and never commit it.

Never “approve” a VM measurement merely by copying a value observed from the
same host during the run.

## Run one stage at a time

### `00-verify-host.sh`

```bash
./00-verify-host.sh
```

Checks the OS, root filesystem, SNP/TDX capability, KVM, IOMMU, GPU model and
driver ownership, swap, RAM, disk, and required network endpoints. Continue
only when the summary reports zero failures. An unbound GPU is normal here.

### `10-install-kubernetes.sh`

```bash
./10-install-kubernetes.sh
```

Installs pinned containerd, CNI plugins, Kubernetes, Helm, and Calico. It
creates a single-node kubeadm cluster, disables swap, uses systemd cgroups,
pins containerd to `/usr/bin/runc`, and makes the control-plane node schedulable.

Expected final line:

```text
Kubernetes is ready
```

### `20-install-coco-gpu.sh`

```bash
./20-install-coco-gpu.sh
```

Installs Kata, the Nydus proxy snapshotter, encrypted confidential-`emptyDir`
support, and NVIDIA GPU Operator in Kata passthrough mode. It waits for the
RuntimeClass, CC-ready label, virtual GPU resource, and `vfio-pci` binding.

On TDX it also installs and validates PCCS/QGS and Intel PCS access. Reboot and
rerun this stage if GPU CC Manager says a reboot is required.

Expected final line:

```text
Kata CoCo runtime and GPU passthrough are ready
```

### `30-deploy-security-services.sh`

```bash
./30-deploy-security-services.sh
```

Creates the Cosign key pair and private registry CA, deploys a TLS registry and
Trustee KBS, installs the registry CA for containerd, and creates a
deny-by-default image-signature policy. It also stages optional NVFlare
hardware-authorizer credentials. For the NVFlare path on TDX, set
`NVFLARE_TDX_ATTESTATION_CONFIG_FILE` to an Intel Trust Authority `config.json`
before this stage; a PCS subscription key is not a substitute. Optionally set
Stage 50 defaults to local NVAT verification. Set `NVFLARE_GPU_VERIFIER=remote`
and `NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE` when using authenticated NRAS.

The infrastructure images are digest-pinned. The unauthenticated registry
binds only to `REGISTRY_BIND_ADDRESS` (normally `NODE_IP`) but permits
push/delete, so firewall port 5000 to trusted administrators.

Back up `state/security/cosign.key` securely. Do not commit or print it.

Expected final line:

```text
Registry and Trustee KBS are ready
```

### `40-sign-image.sh`

```bash
./40-sign-image.sh
```

Copies the configured source image into the local registry, resolves its
immutable digest, signs that digest, verifies the signature, and writes
`state/signed-image.env`.

To choose another image and target tag:

```bash
./40-sign-image.sh docker.io/library/ubuntu:24.04 test-signed
```

### `50-launch-and-attest.sh`

```bash
./50-launch-and-attest.sh
```

Launches a Kata confidential VM with the signed image and one passthrough GPU.
It verifies:

- the measured, deny-by-default Cosign image policy;
- the SNP report and AMD VCEK chain, or TDX quote and Intel PCS collateral;
- approved VM launch measurements and a fresh challenge;
- NVIDIA GPU nonce, certificate, OCSP, RIM, secure-boot, debug, signature, and
  measurement claims; and
- an ephemeral LUKS2/dm-crypt plus dm-integrity confidential `emptyDir` using a
  write/read probe.

The volume is encrypted scratch space, not persistent storage. A successful pod
is deleted by default so the GPU is available for another run. A failed pod is
left in place for diagnostics.

Reports are linked from:

```text
state/latest-attestation-reports
```

Check them with:

```bash
cd state/latest-attestation-reports
sha256sum -c SHA256SUMS
grep -H 'Overall .* result: PASS' *-report.txt
```

## Run everything

After reviewing the configuration and understanding the changes:

```bash
./run-all.sh
```

This runs the generic stages 00 through 50 and stops at the first error.

## Try an NVFlare server and client instead

Install this checkout for the host-side provisioning CLI, then run these exact
NVFlare workflow filenames in order:

```bash
cd ../../..
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[K8S]' tensorboard
cd examples/devops/CoCo
./00-verify-host.sh
./10-install-kubernetes.sh
./20-install-coco-gpu.sh
./30-deploy-security-services.sh
./40-nvflare-build-sign-images.sh
./50-nvflare-deploy.sh
./60-nvflare-submit-hello-numpy.sh
```

This builds separately labeled server/client parent images containing pinned
SNP, TDX, and NVIDIA GPU attestation dependencies, signs and verifies their
immutable digests, provisions `CCBuilder` hardware authorizers, and exposes the
guest-local CPU attestation device inside both Kata VMs. The server issues CPU
evidence; the sole GPU client issues CPU/GPU evidence; both verify peer tokens.
Stages 30 and 50 require a dedicated NVFlare namespace carrying the suite's
ownership label; they refuse to write credentials or enable privileged Pod
admission in an existing unlabeled namespace.
On TDX, configure the Intel Trust Authority file described in the full guide
before Stage 30. Only the client requests `nvidia.com/pgpu`.
Inspect
`state/nvflare-image-signing-report.txt` and
`state/nvflare-deployment-integrity-report.txt`, then require Stage 60 to report
`FINISHED:COMPLETED` for the one-client `hello-numpy` job. The complete
image-integrity, remote-attestation, encrypted-workspace, and job-submission
requirements are in
[`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md).
The generic `40-sign-image.sh`, `50-launch-and-attest.sh`, and `run-all.sh` are
not part of this NVFlare sequence. Do not select scripts with a wildcard.

## Script map

| Script | Short description |
|---|---|
| `run-all.sh` | Runs stages 00 through 50 in order. |
| `00-verify-host.sh` | Validates host, TEE, GPU, storage, and network prerequisites. |
| `10-install-kubernetes.sh` | Installs containerd, Kubernetes, Calico, CNI, and Helm. |
| `20-install-coco-gpu.sh` | Installs TDX quote services when needed, Kata/Nydus, GPU Operator, VFIO, and encrypted-emptyDir support. |
| `30-deploy-security-services.sh` | Creates signing/TLS material, deploys the registry and Trustee KBS, and stages optional NVFlare TDX/GPU authorizer credentials. |
| `40-nvflare-build-sign-images.sh` | Alternate stage 40: builds NVFlare server/client images with pinned SNP/TDX/GPU attestation tools, runs in-image checks, pushes, digest-pins, signs, verifies, and authorizes both repositories. Requires the explicit shared stage 30 first. |
| `40-sign-image.sh` | Copies, digest-pins, signs, verifies, and records the workload image. |
| `50-nvflare-deploy.sh` | Alternate stage 50: provisions CPU/GPU authorizers, exposes `/dev/sev-guest` or `/dev/tdx_guest` inside the Kata VMs, deploys the digest-pinned server/client, and requires the peer hardware-authorization handshake. |
| `60-nvflare-submit-hello-numpy.sh` | Alternate stage 60: exports and submits one-client `hello-numpy`, waits for `FINISHED:COMPLETED`, and prints bounded job logs. |
| `50-launch-and-attest.sh` | Launches the CoCo VM and verifies CPU, GPU, image, and encrypted storage. |
| `lib/common.sh` | Shared configuration, download, checksum, sudo, kubectl, and Helm helpers. |
| `lib/deploy-coco-attest.sh` | Internal stage-50 pod deployment, evidence collection, and verification driver. |
