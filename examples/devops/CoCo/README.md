# Bare-metal CoCo deployment and attestation

This directory is a restartable, staged path from a fresh Ubuntu AMD SEV-SNP or Intel TDX host with a supported NVIDIA confidential-computing GPU to verified CPU, GPU, signed-image, and encrypted-storage reports from a Confidential Container. `TEE_PLATFORM=auto` selects the matching Kata runtime and host checks.

The pinned defaults reproduce the installation validated on this machine: Kubernetes 1.34.9, containerd 2.2.2, Calico 3.32.1, Kata Containers 3.29.0, NVIDIA GPU Operator 26.3.1, Cosign 2.6.2, Trustee KBS, and NVIDIA NVAT 1.2.2.

Documentation:

- [`SUMMARY.md`](SUMMARY.md) is the short trial procedure.
- [`USER_GUIDE.md`](USER_GUIDE.md) is the detailed host and Kubernetes administrator guide.
- [`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md) covers the alternative NVFlare Stage 40/50 path that builds, signs, verifies, and deploys a server and GPU client as confidential VMs after shared Stage 30.

## Before running

Firmware must already have the IOMMU and either AMD SVM/SEV/SEV-SNP or Intel TDX enabled. The NVIDIA GPU must support confidential computing and must not be used by a host NVIDIA driver. These are firmware/hardware conditions the scripts can diagnose but cannot safely change.

Use a dedicated or disposable single-node host. The Kubernetes stage disables swap, installs packages and binaries, writes systemd/containerd/sysctl configuration, and initializes a cluster. Restoring a trusted baseline requires reimaging the host or a separately reviewed administrator-run recovery procedure.

Copy the example configuration and review it:

```bash
cd examples/devops/CoCo
cp config.env.example config.env
editor config.env
```

An empty `NODE_IP` is derived from the default IPv4 route. The registry certificate is issued to that IP, so set it explicitly when the selected address is not stable. Keep the whole directory on local storage: generated keys, certificates, signed-image state, and reports are placed under `state/`.

Intel TDX additionally uses local PCCS/QGS to obtain the platform PCK certificate and generate a TD Quote. Intel's production PCS accepts a request with the subscription header omitted; stage 20 applies a guarded compatibility fix for PCCS 1.21, which otherwise sends an empty header that PCS rejects. If the selected PCS environment requires a key, put the 32-character key in a root-readable file and set `INTEL_PCS_API_KEY_FILE`, or supply `INTEL_PCS_API_KEY` only in the stage-20 environment. Never commit the key. Canonical currently publishes its TDX attestation packages only for selected Ubuntu suites; `TDX_ATTESTATION_PPA_SUITE` is an explicit compatibility override, not an automatic cross-release fallback.

The scripts do not require any pre-existing file under `/tmp`. Host scratch directories are created with `mktemp` for the duration of one stage. Versioned external artifacts are downloaded atomically into `state/downloads/`, reused only when present, and checksum-verified where the publisher provides a pinned checksum. A missing or invalid cached artifact is downloaded again. Files used under `/tmp` inside the CoCo guest are explicitly uploaded or generated and checked before use.

The exact downloaded objects, audited hashes, and publisher checksum sources are listed in [`CHECKSUMS.md`](CHECKSUMS.md). Archive checksums must not be replaced with hashes of binaries extracted from those archives.

Checksum mismatches stop the workflow by default. For exceptional recovery, set `IGNORE_CHECKSUM_MISMATCH=1` in `config.env` or for one invocation, for example `IGNORE_CHECKSUM_MISMATCH=1 ./10-install-kubernetes.sh`. The script prints a warning for every ignored artifact. This disables an important supply-chain integrity check and should not be used for routine installation.

`COSIGN_TLOG_MODE=disabled` is the default for the local private-registry workflow. It verifies the image against the configured Cosign public key but does not claim public transparency or auditability; stages 40 and 50 print an explicit notice. Set `COSIGN_TLOG_MODE=rekor` before running stage 40 to upload the signature to the public Rekor transparency log and require tlog verification in stage 50. Rekor mode requires Sigstore network access and publicly discloses signature metadata for the image reference/digest.

## One-command path

```bash
./run-all.sh
```

The scripts request sudo when a host or cluster-admin change is required. They stop at the first failed prerequisite or deployment check. If a reboot is required to finish GPU confidential-computing mode changes, reboot and rerun `20-install-coco-gpu.sh`, then resume the later stages.

## Individual stages

1. `./00-verify-host.sh` detects SNP or TDX and checks the matching CPU/KVM capability, root-filesystem writeability and ext4 error state, IOMMU groups, absence of conflicting host NVIDIA/nouveau drivers, memory, disk, swap, OS, and required network endpoints. An unbound GPU is valid at this stage. A `FAIL` blocks the workflow; a `WARN` is advisory.
2. `./10-install-kubernetes.sh` installs the host tools, containerd, CNI plugins, Kubernetes, Helm, and Calico; initializes a single-node kubeadm cluster only when one is absent.
3. `./20-install-coco-gpu.sh` installs Kata from its official OCI chart and GPU Operator with Kata sandbox, CC Manager, VFIO Manager, and passthrough device plugin enabled. On TDX it also installs and validates PCCS/QGS, supports production-PCS keyless mode, and validates any supplied PCS key. It waits for the Kata host files, makes the Kata fragment part of containerd's effective configuration, restarts containerd, and verifies the platform-specific `kata-qemu-nvidia-gpu-{snp,tdx}` handler, `nydus-for-kata-tee` plugin, and released `block-encrypted` confidential-emptyDir mode before continuing. It then waits for the RuntimeClass, CC-ready node label, and `nvidia.com/pgpu` capacity, and requires every detected NVIDIA VGA/3D GPU to be bound to `vfio-pci`.
4. `./30-deploy-security-services.sh` creates a private CA, TLS registry, Cosign key pair, deny-by-default image policy, and a Trustee KBS deployment containing verification resources. It also stages optional NVFlare TDX/GPU authorizer credentials; the NVFlare path on TDX requires its Trust Authority configuration here.
5. `./40-sign-image.sh` copies the configured CUDA image into the local registry, signs its immutable digest, verifies it, and writes `state/signed-image.env`.
6. `./50-launch-and-attest.sh` launches the digest-pinned image under the selected SNP/GPU or TDX/GPU Kata RuntimeClass. The guest image pull is accepted only by its measured, deny-by-default Cosign policy. The Pod receives a released CoCo confidential `emptyDir`; Kata creates a host-backed sparse device and applies LUKS2/dm-crypt plus dm-integrity with an ephemeral key generated inside the guest. Stage 50 verifies the mapper, performs a read/write probe, and then collects and verifies SNP or TDX CPU evidence plus NVIDIA GPU evidence. A successful validation pod is deleted so the passthrough GPU is available to reruns; set `KEEP_ATTESTATION_POD=1` only for manual guest inspection.

## Alternate NVFlare workload path

To run an NVFlare server and GPU client instead of the generic stage-40/50
sample, run these exact filenames in order:

```bash
./00-verify-host.sh
./10-install-kubernetes.sh
./20-install-coco-gpu.sh
./30-deploy-security-services.sh
./40-nvflare-build-sign-images.sh
./50-nvflare-deploy.sh
./60-nvflare-submit-hello-numpy.sh
```

Stage 30 explicitly deploys the registry, signing material, and Trustee. Stage
40 builds NVFlare plus pinned SNP, TDX, and GPU attestation dependencies into
distinctly labeled server/client parent images, checks them, pushes them, signs
their immutable digests, and installs a deny-by-default policy. Stage 50 uses
`CCBuilder`, exposes the guest-local CPU attestation interface inside each Kata
VM, and deploys a CPU-evidence server plus CPU/GPU-evidence client. Only the
client requests the passthrough GPU; both verify peer CPU/GPU evidence. Stages
30 and 50 refuse to adopt an existing NVFlare namespace without the suite
ownership label before writing credentials or enabling privileged Pod
admission. TDX users
must stage an Intel Trust Authority config through Stage 30 as described in
[`NVFlare_on_CoCo.md`](NVFlare_on_CoCo.md). That guide contains the complete trust model,
configuration, administrator workflow, and limitations. Stage 60 submits the
standard one-client `hello-numpy` job through an mTLS administrator tunnel and
requires `FINISHED:COMPLETED`. `run-all.sh` does not run this alternate path. Do
not run both stage-40/50 variants or use a shell glob to select the NVFlare
sequence.

To sign a different image or tag without editing the configuration:

```bash
./40-sign-image.sh docker.io/library/ubuntu:24.04 my-signed-tag
```

The signing key is `state/security/cosign.key`; its public verification key is `state/security/cosign.pub`. Set `COSIGN_PASSWORD` in the environment if the key was generated with a password. Never copy, print, or commit the private key.

## Reports and pass criteria

The latest result is linked at `state/latest-attestation-reports`. Important files are:

- `cpu-attestation-report.txt`: decoded platform claims and verification results. SNP includes nonce, `HOST_DATA`, AMD VCEK chain, and ECDSA P-384/SHA-384 results; TDX includes nonce, MRCONFIGID, MRTD/RTMR appraisal, debug state, quote signature, Intel PCS collateral, TCB status, and CRLs.
- `cpu-attestation-report.bin`: the raw 1,184-byte SNP report or raw Intel TD Quote.
- `gpu-attestation-report.txt`: human-readable NVAT result, nonce match, certificate/OCSP, report signature, RIM, secure-boot, debug, and measurement checks.
- `gpu-attestation-evidence.json` and `gpu-attestation-result.json`: raw GPU evidence and verified claims.
- `image-verification-report.txt`: the signed-image policy and measured-init-data binding result.
- `storage-verification-report.txt`: the released CoCo confidential-emptyDir mapper, LUKS2/dm-crypt, dm-integrity, capacity, and read/write verification result.
- `SHA256SUMS`: hashes of all material report artifacts.

Success requires the CPU, GPU, image, and confidential-storage reports to say `PASS`. The CPU challenge is generated fresh for every run and must match SNP `REPORT_DATA` or TDX `REPORTDATA`. The SHA-256 of the init-data containing the image policy, Cosign public key, and private registry CA must exactly match SNP `HOST_DATA` or the first 32 bytes of TDX `MRCONFIGID`.

Stage 50 also requires `EXPECTED_SNP_LAUNCH_MEASUREMENT` to contain an approved 48-byte SNP launch measurement. It verifies the AMD VCEK chain and report signature before comparing the signed `MEASUREMENT` claim with that reference, and fails closed on a mismatch. The pinned example value is the measurement validated for this suite's Kata 3.29.0 `kata-qemu-nvidia-gpu-snp` VM build. Re-establish it from an independently trusted build or release manifest whenever the Kata guest kernel, firmware, rootfs image, dm-verity parameters, hypervisor, or launch configuration changes; never learn or accept a replacement value from the untrusted live host during the same attestation run. A trusted automation system can override the file value with the `EXPECTED_SNP_LAUNCH_MEASUREMENT` environment variable.

On TDX, stage 50 verifies the TD Quote and Intel collateral with a checksum-pinned `google/go-tdx-guest` verifier. It fails closed until `EXPECTED_TDX_MRTD` and all four `EXPECTED_TDX_RTMR*` values are supplied from an independently trusted build manifest or appraisal policy. It also rejects debug-enabled TDs and verifies `MRCONFIGID` against measured init-data. A value merely observed on the same live host is useful for diagnostics, not a trusted reference.

For complete VM-image verification, the launch measurement must be compared against a trusted reference—typically through Trustee/RVPS policy—or enforced with a signed SNP ID block. CoCo's [reference-value architecture](https://confidentialcontainers.org/docs/attestation/reference-values/) is specifically designed for that comparison. This reference deployment performs direct SNP `MEASUREMENT` or TDX MRTD/RTMR comparison in its local verifier; protect the configured references and move appraisal outside the workload host's failure domain for a production threat model.

The confidential volume is ephemeral scratch storage, not an encrypted persistent volume. Its key and detached LUKS2 header exist only in TEE-protected guest memory, and Kubernetes removes its sparse backing file with the Pod. LUKS2 integrity detects individual-block modification but does not prevent a malicious host from replaying an older whole-volume state. The current Kata implementation sizes the logical sparse device from the host filesystem rather than `emptyDir.sizeLimit`; kubelet enforces the requested limit based on physically allocated blocks, so `CONFIDENTIAL_VOLUME_SIZE` must leave enough room for ext4 and integrity metadata. See the [Confidential EmptyDir documentation](https://confidentialcontainers.org/docs/features/protected-storage/confidential-emptydir/).

## Trust and production notes

This is a reproducible bare-metal/reference deployment, not a production PKI design. The local registry uses a private CA, binds only to `REGISTRY_BIND_ADDRESS`/`NODE_IP`, but intentionally has no client authentication and permits push/delete. Restrict its port to trusted administrators with host/network firewall policy. Confidential pulls still fail closed on digest and Cosign policy. Cosign uses key-pair signing; public transparency is optional through `COSIGN_TLOG_MODE=rekor` and disabled by default for the private-registry topology. Trustee KBS is exposed as in-cluster HTTP with an insecure token key, matching the current local validation topology. In production, put the registry and KBS behind authenticated TLS, use a protected signing key (KMS/HSM or keyless identity), enable an appropriate transparency/audit mechanism, define restrictive attestation/resource policies, persist and back up KBS state, and operate trust services outside the workload host's failure domain.

Trustee is the project and KBS is its key-broker service; they are not two separate servers in this deployment. The sample puts the public policy/key/registry CA directly in TEE-measured init-data so signature enforcement is available during the earliest image pull. The same resources are also provisioned into Trustee KBS for later secret/resource retrieval workflows.

## Reruns

Stages use `apply` or `helm upgrade --install` and can be rerun. Existing kubeadm state is never reset. Keys and certificates are reused. To rotate them intentionally, set `ROTATE_SECURITY_MATERIAL=1`, run stage 30, then re-sign the image and rerun attestation; backups receive a UTC timestamp. After rotation, return the value to `0`.

Stage 30 validates the registry certificate on every run: expiry, certificate/private-key match, CA signature, and an IP or DNS SAN matching `REGISTRY_HOST`. If only the server certificate is stale—for example, after moving the suite to a host with a different `NODE_IP`—it preserves the existing CA and Cosign key and automatically issues a new leaf certificate. Replaced leaf files receive a UTC timestamp. The registry deployment is restarted after its TLS Secret is applied so the running process serves the current certificate.

## Upstream references

- NVIDIA Confidential Containers deployment guide: <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/1.0.0/confidential-containers-deploy.html>
- Kubernetes kubeadm/package installation: <https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/>
- Confidential Containers signed images: <https://confidentialcontainers.org/docs/features/signed-images/>
- Confidential Containers confidential emptyDir: <https://confidentialcontainers.org/docs/features/protected-storage/confidential-emptydir/>
- Trustee deployment: <https://confidentialcontainers.org/blog/2026/02/11/deploy-trustee-in-kubernetes/>
