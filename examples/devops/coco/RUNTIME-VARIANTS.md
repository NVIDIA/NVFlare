# Deploy SNP or TDX, with an optional confidential GPU

The supplied role scripts and NVFlare provisioning support these explicit
targets. No script edits are required to select one. The installed host TEE
stack, authenticated launch contract and secure-services approvals must still
match the selected target. TDX hardware end-to-end validation must be performed
on the intended platform; unit/static tests do not establish that result.

| Target | `cc_cpu_mechanism` | `cc_gpu` | `RUNTIME_CLASS` | GPU allocation |
| --- | --- | --- | --- | --- |
| SNP-only | `amd_sev_snp` | `none` | `kata-qemu-snp` | None |
| SNP+GPU | `amd_sev_snp` | `nvidia` | `kata-qemu-nvidia-gpu-snp` | One `nvidia.com/pgpu` |
| TDX-only | `intel_tdx` | `none` | `kata-qemu-tdx` | None |
| TDX+GPU | `intel_tdx` | `nvidia` | `kata-qemu-nvidia-gpu-tdx` | One `nvidia.com/pgpu` |

`kata-qemu-tdx-gpu` is not an alias. Keep `cc_gpu` explicit: missing values,
YAML null and unknown mechanisms fail closed. Do not reuse the SNP+GPU
measurements or launch contract for a different row.

## Support status and hardware validation

**TDX-only and TDX+GPU are experimental pending successful hardware acceptance.**
The additional implementation follows the existing SNP+GPU framework: separate
trusted rehearsal, approved platform references, signed/encrypted participant
images, InitData-bound workload authorization, and CoCoAuthorizer peer proofs.
It generalizes those paths to explicit CPU/GPU targets and adds Intel-specific
quote verification and measured-boot appraisal. Code review and offline tests
support the expectation that the remaining integration should work on a
compliant, correctly configured TDX CoCo cluster; they are not proof of an
end-to-end deployment or a guarantee that no integration defects remain.

In the recorded CPU-only TDX hardware test on **2026-09-29**, Kubernetes and
the pinned Kata runtime were installed and a real collector quote was obtained.
An earlier PCK-certificate retrieval failure was resolved using Intel's
official local-cache workflow. The subsequent strict verification failed:

```text
TDX verification rejected: Intel DCAP TCB is not UpToDate
```

Intel reported `tcb_status=OutOfDate` and `tcb_status_current=OutOfDate`;
collateral was not expired (`collateral_expiration_status=0`). Diagnostic verification
checked the quote signature, fresh challenge, InitData binding and event-log
consistency before the TCB rejection. The canonical verifier also exited with
status 1 and emitted no approved claims. This blocked the workflow at the
hardware firmware/TCB acceptance boundary, not at a successful application
launch. The tested platform still needed vendor-supported firmware/TCB
remediation and a fresh strict attestation pass. A firmware update alone is
not evidence that this acceptance requirement has been met.

No approved reference was exported from that failed test, and the attestation
requirements were not weakened. Successful encrypted NVFlare execution,
attestation-gated workload-key release, and in-Pod CoCoAuthorizer generation
with peer verification remain **unverified end to end on TDX**, including the
TDX+GPU variant. A working collector or passing unit tests do not imply those
later stages succeeded. The tested failure is not a claim that all TDX hardware
is affected, nor a reason to accept `OutOfDate` evidence.

Users deploying into their own TDX CoCo-enabled Kubernetes cluster must first
satisfy the host prerequisites below and complete
[real-hardware acceptance](#6-validate-on-real-hardware). We will investigate
and fix implementation issues encountered and reported during those deployments.
Include software versions and sanitized failing-stage/error details in reports;
do not publish image keys, credentials, raw AA private-key responses, or private
platform evidence. Keep strict appraisal and release policies enabled while
investigating failures.

## CPU-only deployments

CPU-only is an explicit supported target for both protected clients and servers,
not a fallback when GPU attestation fails. Select `cc_gpu: none` in each
participant's CC YAML and use `kata-qemu-snp` or `kata-qemu-tdx` consistently in
the trusted rehearsal, provisioning-node platform configuration and CoCo kit.
Set `GPU_COUNT=0` in the host bootstrap configuration. The source workload Pod
must omit GPU requests/limits; the generated release must not allocate
`nvidia.com/pgpu`.

No NVIDIA GPU, GPU Operator, CUDA userspace libraries or GPU smoke test is
required by this deployment path. Application-specific dependencies remain the
workload owner's responsibility. Skip the optional
[GPU availability diagnostic](trusted_system/GPU-VERIFICATION.md) for CPU-only
profiles. CPU quote/report verification, TCB approval, image signing/encryption,
InitData binding and workload-specific key authorization remain mandatory.
The generated CPU-only KBS policy requires exactly `cpu0`; GPU releases still
require both `cpu0` and `gpu0`. Changing between these modes requires a new
approved launch profile and workload release, not editing an existing Pod YAML.

## 1. Prepare the trusted and untrusted hosts

Use the pinned role kits and [configuration procedure](CONFIGURATION.md).
The trusted platform system is controlled by the platform authority; CoCo IT
controls the adversarial workload cluster. Only the trusted system collects
approval inputs. Sharing a runtime name does not make CoCo IT trustworthy.

SNP hosts need enabled SEV-SNP and the approved host KVM/firmware stack. TDX
hosts need enabled TDX, a compatible kernel/QEMU/Kata stack, Intel platform
provisioning, and a functioning Quote Generation Service. These scripts do not
install firmware, select a security baseline or enroll the host with Intel.

For TDX, independently install and review the host vendor's supported DCAP/QGS
stack. In the trusted-system and CoCo bootstrap configuration, pin:

```bash
TEE_PLATFORM=tdx
RUNTIME_CLASS=kata-qemu-tdx
GPU_COUNT=0
QGS_PACKAGE_VERSION='<reviewed exact installed tdx-qgs package version>'
QGS_CONFIG_SHA256='<reviewed SHA-256 of /etc/qgs.conf>'
QGS_QCNL_CONFIG_SHA256='<reviewed SHA-256 of /etc/sgx_default_qcnl.conf>'
```

Choose the GPU runtime and `GPU_COUNT=1` for TDX+GPU. Review configuration
files and installed package provenance before setting their pins; hashing
unreviewed files is not approval. The QGS daemon must be active. The normal
pinned runtime connects to host VSOCK CID 2, port 4050; if the approved runtime
configuration selects port zero, its Unix QGS socket must instead be available
at the reviewed path. Preflight checks readiness/pins; only a verified fresh
quote demonstrates that this path works.

GPU targets additionally need a supported NVIDIA GPU in production CC mode,
isolated for confidential passthrough. CPU-only targets do not install GPU
Operator, require a GPU device, or allocate a pGPU. Installer filenames
containing `gpu` are retained entry points and dispatch using the selected
configuration.

## 2. Collect an approved platform reference on the trusted system

Use a new private `PLATFORM_PROFILE` and the actual application's reviewed
source Pod shape. Set its `runtimeClassName` from the table. CPU-only Pods
omit GPU resources; GPU Pods request exactly one pGPU. Retain the explicit
application security context, one container, omitted CPU/memory requests and
limits, no host namespaces and no volumes. For NVFlare approve UID/GID 65532
and writable guest-local rootfs; collector diagnostic privileges are not
application permissions.

Follow the common ten-stage order in [trusted_system/README.md](trusted_system/README.md).
For SNP use [the SNP rehearsal runbook](trusted_system/SEC-SYS-LAUNCH-PROFILE.md).
For TDX use [the TDX rehearsal runbook](trusted_system/TDX-LAUNCH-PROFILE.md).
The selected Kata TOML is:

| Runtime | Installed configuration basename |
| --- | --- |
| `kata-qemu-snp` | `configuration-qemu-snp.toml` |
| `kata-qemu-nvidia-gpu-snp` | `configuration-qemu-nvidia-gpu-snp.toml` |
| `kata-qemu-tdx` | `configuration-qemu-tdx.toml` |
| `kata-qemu-nvidia-gpu-tdx` | `configuration-qemu-nvidia-gpu-tdx.toml` |

Stage 05 keeps the same three positional inputs: source Pod, installed approved
Kata TOML, output profile. Stage 10 emits two separate public handoffs: one
reference JSON to secure services, and the v4 launch contract to the provisioning
node. Retain quotes/reports, collateral, event logs and actual-launch evidence
privately on the trusted side. No separate confidential VM is required.

## 3. Install the selected references on secure services

The trusted coordinator transfers the reference JSON over an authenticated
channel. Secure services accepts only references its administrator approves,
never self-reported measurements from CoCo IT.

- SNP uses the unchanged [five-field handoff](service/PLATFORM-REFERENCE-VALUES-HANDOFF.md).
- TDX uses the [versioned complete-profile handoff](service/TDX-REFERENCE-VALUES.md).

A fresh service runs stages 01–11 in order, including stage 02 with
`--configure-only`, and target-specific explicit approval for stage 09.
An existing service adding the new CPU policy must review/install that policy
first. Later reference-only changes use **02 → 10 → 11**, followed by the
documented RVPS restart/readback test. Do not rerun deployment stage 05 for a
reference update: it resets workload release authorization to default-deny.

TDX tuples are stored together in `coco_tdx_profiles_v2`. A later TDX install
replaces that complete approved profile set; include every profile that must
remain accepted. It does not replace SNP approvals. Do not merge individual
measurement fields from different profiles. CPU approval does not authorize
any image key; workload release is a separate stage.

## 4. Build and provision on the NVFlare provisioning node

For each target/profile, prepare a separate admin kit configuration and
authenticate its [approved launch contract](admin/APPROVED-LAUNCH-PROFILE.md).
Set its immutable `RUNTIME_CLASS` to the matching table row. Keep the same
registry trust, signature/encryption requirements and secure-services TLS.

For a TDX-only client, the beginning of its `cc_site-1.yml` is:

```yaml
compute_env: confidential_containers
cc_cpu_mechanism: intel_tdx
cc_gpu: none
role: client
```

Use `cc_gpu: nvidia` for TDX+GPU, `amd_sev_snp` for SNP, or `role: server`
for a protected server. Keep the rest of the reviewed example's image-build,
CCManager and key configuration from [provision/README.md](provision/README.md).
Set top-level `platform_config` to that participant's prepared admin
`platform.env`. Keep the supplied image runner:

```yaml
packager:
  path: nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager
  args:
    build_image_cmd: ../admin/build_coco_image.sh
```

Run from the project directory in the reviewed NVFlare environment:

```bash
nvflare provision -p project.yaml
```

The runner builds the image including the signed participant startup kit,
encrypts all layers, signs the immutable manifest, generates strict policy and
target-aware InitData, and creates the handoffs. Mismatched runtime/GPU request,
launch contract or resources fails closed. Both protected clients and servers
are supported. CPU-only servers need no GPU allocation; a GPU profile still
requires GPU appraisal even when the application does not execute CUDA work.

First send each confidential six-file `trusted-service/` handoff, including
its image key, to the secure-services administrator. Authenticate its manifest
digest and follow [stage 12](service/TRUSTED-HANDOFF-RUNBOOK.md). Only after
approval send the single final Pod YAML and independently authenticated SHA-256
to CoCo IT. Do not send plaintext images, image keys or publisher credentials.

## 5. Install and launch on CoCo

Use [coco/README.md](coco/README.md) with the same selected target. Set the same
`RUNTIME_CLASS` in its `platform.env` and bootstrap configuration. In bootstrap
also set `TEE_PLATFORM`, `GPU_COUNT`, and TDX QGS pins where applicable. Run
**00 → 10 → 20 → 30 → 35 → 40 → 60**. The public chart must arrive before
stage 30; registry CA trust must be installed before launch. Keep secure
services outside this cluster. Do not collect approval references or run
trusted-system stages on this adversarial host.

After receiving the approved Pod and independent digest:

```bash
./50-launch-handoff.sh RELEASE-pod.yaml EXPECTED_SHA256
```

Then use stage 70 as documented in the CoCo role runbook. Do not add
exec/debug/log permissions to diagnose an application. CoCo's launcher is an
operational guard, not the trust boundary: guest policy and independent KBS
authorization enforce integrity.

```bash
./70-verify-running-workload.sh RELEASE-pod.yaml EXPECTED_SHA256
```

## 6. Validate on real hardware

After a fresh authorized launch, run the secure-services release check with
the target's authorization document and a log window containing that launch.
For example, on secure services:

```bash
./13-verify-workload-release.sh RELEASE 5m \
  "$HOME/incoming/RELEASE/release-authorization.json"
```

Confirm CPU appraisal and all three exact resource releases, and GPU appraisal
when required. Confirm the protected participant can generate a CoCoAuthorizer
proof and an ordinary non-CoCo server can verify it using the pinned AS key.
The log inspection is diagnostic, not cryptographic correlation of separate
lines to one attestation session; preserve additional application/proof evidence.

Use authenticated application success as execution evidence, not merely Pod
`Running` state. Test changed command/image/policy, wrong InitData, wrong CPU
type, missing GPU for a GPU release, unapproved or mixed reference tuples and
wrong resource paths. They must fail without approving the negative-test inputs.
Verify references persist across an RVPS restart and existing SNP releases still
work. See [the implementation acceptance matrix](RUNTIME-PORTING.md#acceptance-evidence).

No real TDX end-to-end result is implied by these instructions or by passing
unit tests. Retain dated private acceptance evidence for the actual hardware,
software pins and service policies before declaring a deployment validated.
