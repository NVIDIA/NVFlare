# TDX: collect, verify and approve a workload launch profile

Run this procedure only on the trusted platform system, not the adversarial
CoCo host. It uses a short-lived Kata collector Pod, not a separately managed
confidential VM. The platform security authority approves the software and
security baseline; successful collection does not automatically approve the
reported platform. No real TDX execution is implied by this runbook. Record
the actual deployment's successful verification and acceptance evidence.

## 1. Prepare the host and private inputs

Complete [configuration](../CONFIGURATION.md#3-trusted-platform-system-configuration).
Use a trusted Ubuntu host with TDX firmware/kernel/KVM and Intel platform
provisioning already configured. Install the reviewed Intel DCAP `tdx-qgs`
package and active `qgsd`, then record its exact package version and reviewed
configuration hashes using [the TDX prerequisites](../RUNTIME-VARIANTS.md#1-prepare-the-trusted-and-untrusted-hosts).
Quote collateral acquisition must work through the reviewed QCNL/PCCS path.
Never turn off TLS verification or accept unknown collateral to pass rehearsal.

In the private trusted-system configuration and its bootstrap configuration,
select `TEE_PLATFORM=tdx`, the matching `RUNTIME_CLASS`, and `GPU_COUNT=0` or
`1`. Set the same `QGS_PACKAGE_VERSION`, `QGS_CONFIG_SHA256` and
`QGS_QCNL_CONFIG_SHA256` in **both** private configuration files: stage 04 reads
the main platform configuration, while bootstrap reads its own configuration.
Use a new `PLATFORM_PROFILE`,
private work directories, `PLATFORM_REFERENCE_SKIP_SIGNING=1`, and a stable
absolute `REHEARSAL_WORKLOAD_YAML`. Keep the source unchanged through export.

For TDX-only the source Pod uses `kata-qemu-tdx` and no GPU resources. For
TDX+GPU it uses `kata-qemu-nvidia-gpu-tdx` and exactly one `nvidia.com/pgpu`.
The source application's explicit security context is separately approved;
the collector's diagnostic privileges must not be exported as application
permissions. Use UID/GID 65532 and a writable guest-local rootfs for NVFlare.

From the complete package root:

```bash
CONFIG="$HOME/private-platform-reference/platform-reference.env"
source "$CONFIG"
PROFILE="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
bash trusted_system/01-install-rehearsal-host-tools.sh "$CONFIG"
# Reconnect so Docker group membership is effective, then restore the variables.
bash trusted_system/02-install-kubernetes.sh
bash trusted_system/03-fetch-kata-artifacts.sh "$CONFIG"
bash trusted_system/04-install-rehearsal-runtime.sh "$CONFIG"
```

Stages 03/04 retain immutable runtime/chart pins, derive the token/evidence API
configuration, and select the target TOML. GPU Operator and GPU readiness checks
are skipped for CPU-only; no NVIDIA hardware is required. Do not reuse TDX+GPU
reference evidence for TDX CPU-only: collect and approve each selected profile.
Host readiness or an available QGS socket is not proof of successful attestation.

## 2. Approve launch inputs and the verification baseline

Select the installed configuration matching the runtime:

```bash
KATA_CONFIG=/opt/kata/share/defaults/kata-containers/configuration-qemu-tdx.toml
# For TDX+GPU instead use configuration-qemu-nvidia-gpu-tdx.toml.
python3 trusted_system/05-define-approved-launch-profile.py \
  "$REHEARSAL_WORKLOAD_YAML" "$KATA_CONFIG" \
  "$PROFILE/approved-launch-profile.json"
bash trusted_system/06-prepare-platform-reference.sh "$CONFIG"
editor "$PROFILE/platform-approval.env"
```

Stage 06 builds the source-pinned Trustee/DCAP verifier in a container and
records the verifier path/hash in the private derived configuration. No host
Rust compiler or separately installed DCAP verifier is required. Review its
source/pins and the platform artifacts before accepting its result.

Set `TDX_SECURITY_BASELINE_APPROVED=1` only after the platform authority
approves this strict baseline: verified DCAP quote, `UpToDate` TCB result,
unexpired collateral, debug disabled, measured-boot replay against all four
RTMRs, fresh 64-byte REPORTDATA, and expected InitData/MRCONFIGID binding.
The scripts do not learn a weaker baseline from the machine under test. Do not
set AMD TCB fields for TDX. Leave `APPROVED_TDX_PROFILE_SHA256` blank until the
verified repeat produces the candidate in the next step.

## 3. Collect fresh evidence twice, then approve the candidate

```bash
bash trusted_system/07-run-snp-rehearsal.sh \
  "$CONFIG" "$PROFILE/platform-approval.env"
python3 trusted_system/08-repeat-profile-rehearsal.py "$PROFILE"
python3 -m json.tool "$PROFILE/candidate-tdx-profile.json"
sha256sum "$PROFILE/candidate-tdx-profile.json"
```

The stage-07 filename is historical: it dispatches to TDX for the selected
TDX runtime. The collector accesses the guest-local attestation agent, using
the measured `agent.guest_components_rest_api=all` runtime profile. Host tools
verify the evidence with the pinned verifier; they do not trust a decoded quote
or unverified guest-reported JSON. The actual Pod-associated QEMU launch and
artifact hashes are retained and compared to approved inputs. The repeat must
use a distinct fresh challenge and the same stable reference tuple/launch shape.

The platform authority reviews the verified candidate alongside the actual
launch/artifact evidence. If acceptable, edit the approval file and set
`APPROVED_TDX_PROFILE_SHA256` to the exact SHA-256 printed above. Do not set this
automatically without review; this is the authorization decision, not merely
format validation. Preserve failed evidence and choose a new profile for a
new collection rather than overwriting or mixing runs.

The candidate includes kernel and kernel-parameter event digests as well as
MRTD/RTMR/XFAM. Approve the entire tuple; do not cherry-pick lower or convenient
values from a different report. For TDX+GPU, GPU allocation/readiness during
collection does not replace NVIDIA remote appraisal of the eventual encrypted
workload. CPU-only requires neither GPU collection nor NVIDIA appraisal.

## 4. Finalize and export both handoffs

```bash
bash trusted_system/09-finalize-platform-reference.sh \
  "$CONFIG" "$PROFILE/platform-approval.env"
install -d -m 0700 "$PROFILE/handoff/secure-services" "$PROFILE/handoff/admin"
bash trusted_system/10-export-platform-reference-values.sh \
  "$PROFILE/platform-reference.final.env" \
  "$PROFILE/handoff/secure-services/platform-reference-values.json" \
  "$PROFILE/handoff/admin/approved-workload-launch-profile.json"
```

Stage 09 re-verifies both retained fresh reports and collateral, their distinct
challenge bindings, measured boot, unchanged source/artifacts, actual launch
and the explicitly approved candidate hash. Stage 10 refuses existing output
paths. Its TDX service document uses `coco-platform-reference-values/v2`,
`tee: tdx`, and a complete approved profile tuple; the separate admin contract
uses `coco-approved-workload-launch/v4`. The two files are not interchangeable.

The provisioning node coordinates authenticated delivery: send only the
reference JSON to secure services and the separately authenticated launch
contract to the workload owner. Follow [TDX service approval](../service/TDX-REFERENCE-VALUES.md)
and [admin contract installation](../admin/APPROVED-LAUNCH-PROFILE.md).
Keep reports, collateral, logs, verifier records and private evidence on the
trusted side. CoCo IT receives only the public runtime inputs and, later, the
workload owner's final Pod YAML. No signed runtime bundle is required.

## Verification limits and final acceptance

The collector verifies the CPU launch and approved profile; it does not run
the encrypted application or prove decryption-key release. After service
approval, provision a real workload, launch its unchanged Pod on the selected
runtime, verify key release and any required NVIDIA appraisal, and have an
ordinary server verify the guest's CoCoAuthorizer proof. Test deliberate
measurement/InitData/policy substitutions and confirm denial. Preserve those
results before claiming end-to-end TDX validation.
