# Collect five references using the workload's trusted launch profile

This procedure runs only on the trusted platform system. It does not configure
the adversarial cluster or secure services. The platform owner must approve the
AMD firmware baseline. The four verified reported TCB values become minimum
floors; evidence collection is not an independent firmware vulnerability audit.

## Scope and approved inputs

The source workload is `example-workload-v1-pod.yaml`. Its relevant launch
conditions are one container, RuntimeClass `kata-qemu-nvidia-gpu-snp`, one
`nvidia.com/pgpu`, no CPU/memory requests or limits, and no host namespaces.
Kata supplies the omitted sizing: 1 vCPU and 8192 MiB. Kubernetes defaults the
GPU request from its limit. The approved profile records these fields, the
host CPU identity, full Kata configuration, and hashes of the QEMU executable,
firmware, kernel and rootfs. The running guest used QEMU CPU model `EPYC-v4`.

The collector substitutes its image, command, privileged diagnostic context,
and temporary-registry init-data for the workload's own content. It does NOT
run the encrypted application or test application key release. It reproduces
the selected VM resource/device conditions and captures the actual QEMU
launch. CPU report verification is performed; GPU allocation/CC readiness is
checked, but this is not a cryptographic GPU/NRAS attestation test.

Two matching reports establish reproducibility for this tested profile on
trusted_system, not all workloads or another machine. The five values alone do not
authenticate workload init-data, its agent policy, or GPU evidence. Those
remain separate secure-services policy checks. An adversarial host's claimed
configuration or a local checksum is never a substitute for attestation.

## 1. Prepare trusted_system and the configuration

Use the account `operator` with passwordless sudo. Transfer this directory
and the chosen source Pod YAML to trusted_system. Do not clone NVFlare. The Kubernetes
bootstrap files are already included; their provenance is in `bootstrap/README.md`.
First create and review both private configurations as described in
[CONFIGURATION.md](../CONFIGURATION.md#3-trusted-platform-system-configuration).
The installers compare `EXPECTED_HOSTNAME` against this host; the runtime stage
discovers the single NVIDIA GPU PCI address instead of assuming a fixed slot.
Place the reviewed source YAML at the configured `REHEARSAL_WORKLOAD_YAML`.

From the package root:

```bash
CONFIG="$HOME/private-platform-reference/platform-reference.env"
source "$CONFIG"
PROFILE="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
bash trusted_system/01-install-rehearsal-host-tools.sh "$CONFIG"
# Reconnect SSH so Docker group membership is effective; restore these variables.
bash trusted_system/02-install-kubernetes.sh
```

Bootstrap pins Kubernetes 1.34.9, containerd 2.2.2, Helm 3.21.3 and Calico
3.32.1. It installs cri-tools and keeps kubeadm initialization logs private.
No Trustee or persistent registry is installed by host bootstrap.

For a new collection choose a new `PLATFORM_PROFILE`; do not overwrite old
evidence. The source YAML must be present before defining the profile.
`PLATFORM_REFERENCE_SKIP_SIGNING=1` is appropriate for this five-value-only
handoff; no signing key, password, or CoCo bundle is generated.
Signed bundle builders, the standalone bundle recalculator, the cross-role
validator and capture-helper test file have been removed from this kit. The
remaining helper scripts and collector build files are runtime dependencies
and must be retained.

## 2. Acquire, install and define the approved launch profile

```bash
bash trusted_system/03-fetch-kata-artifacts.sh "$CONFIG"
bash trusted_system/04-install-rehearsal-runtime.sh "$CONFIG"
python3 trusted_system/05-define-approved-launch-profile.py \
  "$REHEARSAL_WORKLOAD_YAML" \
  /opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml \
  "$PROFILE/approved-launch-profile.json"
bash trusted_system/06-prepare-platform-reference.sh "$CONFIG"
```

Stage 03 checks the chart archive digest and immutable Kata image. Stage 04
installs that chart and image, restarts containerd/kubelet, and installs GPU
Operator 26.3.1 with confidential-computing management and VFIO passthrough.
It requires SNP enabled, CC-ready GPU status, the SNP RuntimeClass and the
nydus snapshotter. Stage 05 rejects unsupported profiles instead of silently
approving them. Stage 06 prepares the isolated measurement-tool environment
and the approval template. No separate confidential VM is needed.

## 3. Rehearse and independently repeat the launch

```bash
bash trusted_system/07-run-snp-rehearsal.sh \
  "$CONFIG" "$PROFILE/platform-approval.env"
python3 trusted_system/08-repeat-profile-rehearsal.py "$PROFILE"
```

Stage 07 generates a fresh 64-byte challenge, builds a checksum-pinned
`snpguest` collector, and temporarily serves it from a TLS registry on trusted_system.
The Pod retains the source workload's resource settings. While it runs, the
trusted host associates its Pod UID with its CRI sandbox and captures that
sandbox's QEMU process, artifact hashes, CPU model, topology, memory, machine
type and full kernel command line. It fails if the actual launch is missing
or is not SNP. The brief post-report hold allows this capture before exit.

The trusted host verifies the AMD certificate chain, report signature, VCEK
TCB correspondence, and signed challenge binding. It retains the evidence
and automatically records the four TCB floors. Stage 08 is an additional
repeat check: it uses a different fresh challenge and requires another
verified report with the same measurement and TCB, plus identical captured
launch inputs and resource fields. Run it before accepting/exporting this
profile. It is a separate check, not automatically invoked by stage 09.

Both scripts remove their own temporary Pod namespace and registry container.
Evidence, collector images and temporary-registry data/certificates remain
under the private profile. On failure, retain these for diagnosis; choose a
new profile or explicitly move the failed run directory aside before retrying.
Do not bypass evidence verification to obtain an output.

## 4. Finalize and export only the five values

After BOTH rehearsals pass:

```bash
bash trusted_system/09-finalize-platform-reference.sh \
  "$CONFIG" "$PROFILE/platform-approval.env"
install -d -m 0700 "$PROFILE/service-out"
bash trusted_system/10-export-platform-reference-values.sh \
  "$PROFILE/platform-reference.final.env" \
  "$PROFILE/service-out/platform-reference-values.json"
python3 -m json.tool "$PROFILE/service-out/platform-reference-values.json"
```

Stage 09 checks the approved resources/artifacts against the actual launch,
checks the source YAML and capture hashes, and re-verifies the signed report,
challenge and TCB values. The reported measurement is authoritative. The
offline calculation is diagnostic only: it differed in this tested run and
was NOT exported. Stage 10 refuses an existing output and emits exactly five
fields with no keys, certificates, policies, reports or artifact archives.

Only `service-out/platform-reference-values.json` is the secure-services
handoff. Deliver it through an authenticated channel and follow
`../service/PLATFORM-REFERENCE-VALUES-HANDOFF.md`; this procedure does not
automatically transfer it or change that node's RVPS/AS policies.

To also prepare provisioning_node's required workload launch contract, give stage 10
a third output path. Use new output paths rather than overwriting this run:

```bash
install -d -m 0700 "$PROFILE/handoff-with-admin/secure-services" "$PROFILE/handoff-with-admin/admin"
bash trusted_system/10-export-platform-reference-values.sh \
  "$PROFILE/platform-reference.final.env" \
  "$PROFILE/handoff-with-admin/secure-services/platform-reference-values.json" \
  "$PROFILE/handoff-with-admin/admin/approved-workload-launch-profile.json"
```

Stage 09 binds the approved profile and actual launch capture by SHA-256 in
the final environment. The admin exporter refuses changed or unbound evidence.
The secure services output remains exactly five values; the admin contract is separate.
Authenticate and install it as described in
[APPROVED-LAUNCH-PROFILE.md](../admin/APPROVED-LAUNCH-PROFILE.md). This adds no
numbered stage and grants no additional authority to CoCo IT.

## Validation boundary

Collect fresh evidence on the intended approved platform. No report or
measurement from a previous installation is packaged. A successful collector
proves the verified CPU report and recorded launch, not application execution
or GPU remote attestation. Secure services must appraise CPU and GPU evidence
for the actual encrypted workload before KBS releases its resources.
