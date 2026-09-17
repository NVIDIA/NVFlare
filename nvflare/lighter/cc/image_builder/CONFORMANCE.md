# NVIDIA Confidential Computing Reference Architecture Conformance

Review date: September 17, 2026
Original review: commit `eaff81ca0b7a09f6a128aa9bd9f14c76586088ed`.
September 15 follow-up validation is recorded below. The September 17 composite
GPU authorization and PR boundary tests are recorded in [VALIDATION.md](VALIDATION.md).

Reference: NVIDIA *Deploying Proprietary Models in Confidential Compute —
Self-Hosted VMs*, last updated June 4, 2026

## Purpose and scope

This document compares the CVM Builder implementation with the NVIDIA reference
architecture (RA). The builder is application-neutral and supports two distinct
deployment profiles:

- A CPU-only confidential-computing profile (`gpu: none`). This is a valid mode
  for NVFlare and other workloads. GPU evidence is neither available nor required
  for key release in this profile.
- An NVIDIA confidential-GPU profile (`gpu: nvidia_cc`). This profile is within
  the cited RA's GPU-accelerated inference scope. A key protecting a workload that
  requires a confidential GPU must be authorized using both CPU-CVM and GPU
  evidence.

GPU attestation is therefore a policy condition, not a global requirement of the
key service. The key service may hold other keys with CPU-only or other resource
policies. A CPU-only NVFlare deployment must not be rejected because it has no
GPU. Conversely, a deployment must not claim conformance to the cited
GPU-inference RA unless its GPU-dependent keys are gated by successful GPU
appraisal.

The RA's Dell server and eight-GPU configuration is a worked example rather than
a universal hardware mandate. An alternate platform can conform when its exact
hardware and software profile is documented, validated, and approved.

## Status legend

| Status | Meaning |
|---|---|
| Meets | The implementation contains the required control and there is applicable test evidence. |
| Conditional | The control is present when the documented profile or deployment procedure is followed. |
| Gap | The implementation does not yet provide the required control for an applicable profile. |
| Evidence pending | The control exists, but current end-to-end hardware evidence is incomplete. |
| Outside RA scope | Supported builder behavior that is not a claim of conformance to this GPU-inference RA. |

## Overall assessment

The builder implements most of the RA's core security mechanisms: measured SNP
and TDX guests, fresh CPU attestation, exact measurement and vault binding,
policy-controlled KBS release, memory-only key handling, authenticated encrypted
vault storage, fail-closed workload startup and periodic re-attestation.

The CPU-only NVFlare profile does not have a GPU conformance gap; it follows the
CPU-CVM trust path and is outside the cited RA's GPU-inference scope.

The GPU-enabled profile now gates vault keys on a composite CPU/GPU EAR at
Trustee. Its implementation is covered by policy, signed-token and HTTPS tests;
physical acceptance of the new composite path remains pending. Full conformance
also depends on the exact production hardware/BOM and remaining deployment
requirements below.

Additional work is required to make a production conformance claim: record the
exact approved deployment profile, restrict network traffic by endpoint as well
as port, complete the operational audit fields and export path, make OCI signature
verification an enforced delivery step, and run the complete acceptance matrix on
freshly built production artifacts.

## September 15 validation follow-up

Fresh CPU-only TDX and H800/SNP candidates were built and round-tripped through
the lab OCI registry. Diagnostics confirmed H800 passthrough after reserving a
256 GiB PCIe window and successful remote NVIDIA cryptographic appraisal through
the pinned C++ `nvattest` client. The current policy handles two schema claims
omitted by NRAS without making report/RIM signature, certificate, nonce, version
or measurement checks optional.

The complete current hardware matrices remain pending. They exposed cold-boot
NTS connectivity and chrony initialization issues; the initialization regression tests now pass on both Linux hosts. All 13
applicable final TDX hardware cases and its zero-argument offline OCI launch
passed. SNP GPU hardware acceptance remains blocked by another Kata workload
holding the physical GPU.
See [VALIDATION.md](VALIDATION.md) for completed test counts, timed build stages,
and the exact limits of the evidence. The [Trustee setup guide](TRUSTEE_GUIDE.md)
adds deployment instructions without changing the CPU-only key-release policy
or establishing hardware acceptance of the new composite key-release path.

## Required capabilities

The requirements in this table come from the RA's
[Required Capabilities](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/required-capabilities.html)
and
[Attestation and Key Release Flow](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/attestation-and-key-release-flow.html).

| RA capability | CPU-only profile | GPU-enabled profile | Implementation evidence and remaining work |
|---|---|---|---|
| Launch the workload in a measured CVM | Meets | Evidence pending | SNP and TDX launch assets and shape are pinned in `builder/launcher.py`; exact reference measurements are generated and approved in `builder/policy.py`. Current source changes require fresh hardware acceptance before production approval. |
| Collect fresh evidence for the CPU TEE, guest image, launch configuration, firmware, and applicable GPU | Meets | Conditional | `builder/attestation.py` obtains a fresh KBS challenge, binds it to an ephemeral key, validates the signed EAR, and checks the selected CPU profile. The patched client collects NVIDIA evidence bound to RCAR runtime data; Trustee verifies NRAS signatures and applies the generated GPU policy. GPU appraisal has not been demonstrated in the complete current CVM flow. |
| Release a protected key only after fresh evidence matches policy | Meets | **Evidence pending** | `authorized_key()` uses a composite RCAR transaction. The CPU quote covers GPU evidence; KBS requires the configured NVIDIA GPU submods before releasing a GPU vault key. CPU-only rules are unchanged. Physical negative acceptance remains pending. |
| Keep keys out of host-visible storage and normal VM-management paths | Meets | Meets | KBS response decryption uses an ephemeral private key; plaintext keys are held in sealed memory file descriptors and passed to `cryptsetup` through `/proc/self/fd`. No plaintext key is written to a disk or command line. |
| Keep confidential model artifacts encrypted outside the CVM | Conditional | Conditional | The vault is LUKS2-encrypted and authenticated. `/user_config`, `/user_data`, and `/applog` are intentionally clear sidecars. Confidential model weights, credentials, and proprietary application material must be placed in the vault rather than those sidecars. The builder rejects obvious private keys in public inputs but cannot infer the confidentiality of arbitrary data. |
| Fail closed when attestation, policy evaluation, key release, integrity validation, or re-attestation fails | Meets | Conditional | Systemd failure handling powers off the guest; the workload target is reached only after vault and integrity gates. Periodic CPU/KBS and GPU checks also fail closed. The composite GPU path still needs full hardware validation. |
| Emit privacy-safe audit records for attestation and key release | Conditional | Conditional | `builder/audit.py` records time, build ID, vault ID, measurements, policy ID, and allow/deny without tokens, keys, or payloads. It lacks a correlation/request ID, verifier identity, key-resource ID, safe failure-reason code, and a demonstrated SIEM/export integration. Trustee/KBS logs may supply some fields, but the deployment must prove the combined record. |

## Architecture and deployment controls

| Area | Status | Finding |
|---|---|---|
| Trust boundaries | Meets | The host assembles and launches the VM but does not receive the vault key. The KBS and appraisal policy authorize exact CPU measurements and a vault binding. The guest verifies the same local binding before contacting the KBS. |
| CPU attestation policy | Meets | SNP policy checks measurement, minimum TCB values, debug/migration, SMT, ABI, and a single-socket profile. TDX policy checks MRTD, RTMR0/1/2, CCEL presence, MRSEAM/TCB, status, collateral, advisories, debug, and XFAM. Token signature keys and algorithms are pinned. |
| GPU attestation policy | Conditional | `config/gpu_policy.json` checks the overall appraisal result, secure boot, debug state, measurement result, certificate/report/nonce/signature results, driver RIM, and VBIOS RIM. Exact GPU count is checked separately. The selected production profile must also identify and validate GPU SKU, firmware, topology, and CC mode. |
| Hardware/software profile | **Gap** | The build manifest pins the guest, firmware input, package versions, CPU model, vCPU count, memory, GPU mode, and GPU count. It does not capture the RA's complete deployment profile: server model, physical CPU, GPU SKU/form factor/topology, BIOS and microcode versions, host kernel/QEMU, NICs or DPU, management network, and profile-specific acceptance results. These may live in a separate signed BOM, but approval must bind it to the released profile. |
| Guest administrative access | Conditional | SSH, console getty, emergency/rescue login, cloud-init state, unlocked passwords, swap, and core dumps are removed or disabled. QEMU uses `-nographic`. The acceptance suite does not yet demonstrate every Appendix E host/admin path, including QEMU monitor access, host-side attach, and memory-dump attempts. |
| Inbound and outbound network policy | Conditional | Guest nftables has default-drop input/output/forward chains and port allowlists. An allowed outbound port such as TCP 443 can reach any destination, and an allowed inbound port accepts any source. The RA limits traffic to approved callers and approved verifier, KBS, and artifact endpoints. A gateway, firewall, or egress proxy with address rules can satisfy this today; without that external control, this is a gap and the builder needs source/destination CIDR or endpoint allowlists. |
| TLS and service authentication | Conditional | KBS and GPU appraisal URLs require HTTPS, and KBS trust is pinned. The generic application container is responsible for its own HTTPS listener, peer authentication, and authorization. A production RA deployment must document those controls. |
| Encrypted storage and binding | Meets | The vault uses detached LUKS2 metadata, AES-XTS data encryption, HMAC-SHA256 authentication, a frozen inspected header, exact device/mapping validation, and a full authenticated scan before workload startup. The unencrypted measured root contains generic software and public policy rather than application secrets. |
| Clear sidecars | Conditional | `/user_config` and `/user_data` are operator-supplied, unencrypted, and read-only in QEMU, the guest, and the container. `/applog` is intentionally unencrypted and writable for operator-readable output. They conform only when their contents are classified as nonconfidential and logs contain no secrets, proprietary payloads, or sensitive evidence. |
| Artifact identity and integrity | Conditional | OCI materialization verifies content digests and registry pulls require digest references. The user guide requires `cosign verify`, but signature/provenance enforcement is outside the launcher. A production release process must make verification mandatory or add it to the materialization wrapper. |
| Image and key lifecycle | Conditional | The generic CVM build and application vault are independently versioned. Approval receipts, KBS resource installation, retirement/revocation, OCI digests, and rebuild/update procedures implement the RA lifecycle model. Current measured artifacts must be rebuilt and reapproved after source changes. |
| Operational signals | Conditional | Local audit records are privacy-safe, but the RA also expects a policy decision, verifier, useful safe failure reason, and operational correlation. Export, retention, alerting, clock monitoring, and SIEM integration remain deployment work. |

The architecture observations above use the RA's
[Architecture Summary](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/architecture-summary.html),
[Hardware and Platform Software Requirements](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/hardware-and-platform-software-requirements.html),
[Network and Service Integration](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/network-and-service-integration.html),
[CVM Image and Model Lifecycle](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/cvm-image-and-model-lifecycle.html),
and
[Operations and Failure Handling](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/operations-and-failure-handling.html).

## Appendix E acceptance matrix

The RA's
[Appendix E acceptance tests](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/appendix-e.html)
are the minimum production evidence set. `VALIDATION.md` records the existing
software and lab results; the table below identifies what remains for the current
source revision.

| RA acceptance test | Status for current implementation | Required evidence |
|---|---|---|
| Positive key release | Evidence pending | Rebuild and approve current SNP and TDX bundles, boot each delivered CVM plus vault, and record successful fresh appraisal, key authorization, integrity scan, and application readiness. For a GPU profile, include successful GPU appraisal before the protected key release. |
| Unapproved image | Conditional | Exact measurement denial is covered by policy and unit tests. Repeat on current hardware with a modified or unapproved image and show that the key is unavailable. |
| Tampered launch parameters | Evidence pending | Test changed vCPU count, CPU model, memory/launch shape, firmware, and measured boot inputs as applicable; show appraisal or binding denial before key release. |
| GPU CC mode disabled or invalid GPU evidence | **Evidence pending for a GPU profile** | Composite authorization is implemented. Run missing/tampered/replayed GPU and CC-disabled hardware cases and prove no key release, no vault mapper, no allow record and bounded poweroff. CPU-only profiles do not run this test. |
| Expired or revoked collateral | Conditional | JWT lifetime, appraisal status, trust vector, and collateral claims are checked in software tests. Add a real expired/revoked attestation collateral case on the production verifier path. |
| KBS/KMS outage | Meets, refresh after rebuild | The recorded TDX KBS DROP test powered the guest off within the acceptance limit. Repeat on the final production artifacts and both selected CPU platforms. |
| Key disable or revocation | Meets, refresh after rebuild | KBS resource deletion/retirement and periodic authorization failure are implemented and tested. Repeat on final artifacts and retain the KBS audit record. |
| Administrator bypass attempts | Evidence pending | Existing tests cover SSH/service/socket/core-dump hardening. Add console, QEMU monitor, host attach, memory-dump, and direct vault-access attempts expected by the deployed management stack. |
| Artifact and key rotation | Conditional | Versioned builds, vaults, resource paths, approval, retirement, and OCI digests support rotation. Demonstrate an old-artifact denial and new-artifact success on the final release. |

## Findings and required actions

### C1 — Gate GPU-dependent keys on GPU appraisal

**Status: Evidence pending.** Applies only to `gpu: nvidia_cc`.

The pinned Trustee and guest client now support composite CPU/GPU evidence in a
single RCAR transaction. CPU REPORT_DATA covers the additional GPU evidence and
all devices share the challenge and ephemeral response key. The backend performs
NRAS verification, verifies both signed JWTs and their digest/nonce linkage, and
emits one GPU submod per distinct NVIDIA device. Immutable CPU/GPU AS policies and
RVPS driver/VBIOS approvals govern the resource rule. Missing, extra, duplicate,
non-NVIDIA, wrong-policy or non-affirming GPU submods deny GPU vault keys.
CPU-only rule bytes are unchanged. Periodic authorization repeats the composite
transaction; the guest only controls CUDA readiness.

Policy and signed-fixture tests do not establish physical conformance. Promotion
to **Meets** requires exact-profile hardware evidence for
`gpu_negative_key_denial`, `gpu_positive_key_release`, `gpu_policy_selection`,
`cross_class_denial` and `periodic_gpu_denial`, including CC-disabled, tampered,
replayed and missing-device cases. Verify no allow record or vault mapper before
key denial and bounded poweroff. Existing hardware evidence predates this change.

### C2 — Bind approval to an exact production profile and BOM

Severity: High for a conformance claim.

Add a signed profile/BOM record containing the exact server, CPU TEE, GPU details
when applicable, firmware, microcode, host kernel and QEMU, guest versions,
network hardware, management boundary, and acceptance result identifiers. Bind
the record's digest to the approved CVM profile or approval receipt. The RA's
[Appendix F](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/appendix-f.html)
provides the expected BOM categories.

### C3 — Restrict traffic to approved endpoints and callers

Severity: Medium.

Define and enforce source address rules for inbound application traffic and
destination address or proxy rules for KBS, verifier, artifact, DNS, and time
services. Until the guest schema supports address constraints, document the
external firewall or gateway that supplies this control and include it in site
acceptance.

### C4 — Complete privacy-safe operational records

Severity: Medium.

Add a correlation/request ID, policy version, verifier identity, key-resource ID,
component ID, and allowlisted failure-reason code. Correlate guest and Trustee
events, export them to the selected log/SIEM system, and test that no key, token,
evidence body, model data, user data, or private configuration appears in logs.
The suggested interface fields are in
[Appendix C](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/appendix-c.html).

### C5 — Enforce signed OCI provenance

Severity: Medium.

Keep digest verification and digest-pinned registry references. Make signature
and provenance verification mandatory in the artifact materialization command or
in a release-channel control that cannot be skipped. Record the verified digest
and signer identity with the approval.

### C6 — Complete fresh hardware acceptance

Severity: High for release readiness.

Rebuild after the latest measured-root and schema changes, approve those exact
artifacts, and run the full matrix above. A GPU-inference conformance claim needs
a complete GPU-enabled CVM run, including GPU passthrough, NVIDIA remote
attestation inside the workload environment, GPU-negative key denial, rotation,
and periodic failure behavior. A standalone GPU attestation container proves SDK
connectivity but does not prove the CVM key-release sequence.

## Conformance boundary

The implementation may be described as providing the core mechanisms for the RA
and as supporting CPU-only confidential NVFlare deployments. It should not yet be
described as fully conformant to the NVIDIA GPU-inference RA. A conformance claim
can be made for an exact deployment profile after C1 through C6 are resolved or
supplied by documented deployment controls and the complete acceptance evidence
passes.

The RA's
[Scope](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/scope.html)
explains that equivalent stacks may conform when they preserve the security
boundaries and required capabilities. Sample Ubuntu, driver, firmware, server,
and GPU-count values in
[Appendix A](https://docs.nvidia.com/enterprise-reference-architectures/deploying-proprietary-models-confidential-compute-self-hosted-vms/latest/appendix-a.html)
are not universal requirements; every selected combination still needs its own
validation and approval.

## Review limitations

This is a source, design, and recorded-evidence review. It does not certify a
deployment, a particular NVIDIA GPU/driver/firmware combination, the external
Trustee configuration, network perimeter controls, log retention, or an artifact
registry. `VALIDATION.md` remains the record of executed tests. Production
conformance is established only by the signed configuration/BOM and acceptance
evidence for the exact delivered artifacts and target site.
