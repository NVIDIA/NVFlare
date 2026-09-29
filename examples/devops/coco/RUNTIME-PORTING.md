# Runtime implementation and security contract

For operator commands use [RUNTIME-VARIANTS.md](RUNTIME-VARIANTS.md), not code
modifications. The supplied scripts select SNP/TDX and CPU-only/GPU targets.
This document records the invariants that future runtime changes must preserve.
Local tests do not establish real TDX quote, key-release or workload execution;
complete the hardware acceptance procedure before claiming those outcomes.

## Target mapping and approved launch inputs

The explicit CPU/GPU pair selects one of four runtime classes:

| CPU | GPU | RuntimeClass |
| --- | --- | --- |
| `snp` | `none` | `kata-qemu-snp` |
| `snp` | `nvidia` | `kata-qemu-nvidia-gpu-snp` |
| `tdx` | `none` | `kata-qemu-tdx` |
| `tdx` | `nvidia` | `kata-qemu-nvidia-gpu-tdx` |

Host bootstrap verifies the selected host stack. GPU Operator, device ownership,
production CC readiness and one-pGPU allocation apply only to GPU targets.
TDX preflight requires a pinned preinstalled QGS and reviewed collateral
configuration; it does not install host firmware, enroll the Intel platform or
infer quote validity from an open socket.

`coco-approved-workload-launch/v4` adds explicit `cpu_tee` and `gpu`, consistent
with `runtime_class`. The contract retains source approval, runtime digest,
actual artifact/configuration hashes, guest token API capability, effective VM
launch inputs, and the approved application security context. Legacy v3 is
accepted only for its original SNP+GPU profile; changing its identifier or fields
does not grant TDX approval. Earlier contracts need a fresh reviewed rehearsal.

The supported workload shape remains one container, no CPU/memory requests or
limits, no host namespaces/volumes, and zero or exactly one `nvidia.com/pgpu`.
Both protected clients and servers use this contract. Arbitrary VM sizing,
multiple GPUs or extra containers need additional implementation and review.

## Verified references, not host assertions

The trusted host creates a fresh challenge, launches a collector in the selected
Kata runtime, correlates the Pod to the actual QEMU process, and verifies the
retained evidence. A repeat rehearsal uses a different challenge and must match
the approved stable measurements and effective launch inputs. Finalization
rechecks evidence and source/artifact bindings before export.

SNP retains AMD report/certificate verification and independently approved TCB
floors. TDX uses the pinned Trustee/DCAP verifier rather than implementing quote
cryptography in the scripts. It requires acceptable quote/collateral status,
nonce and InitData binding, a consistent measured-boot log, and non-debug
configuration. A guest-supplied measurement or successful Kubernetes status
does not satisfy those checks.

The TDX handoff is `coco-platform-reference-values/v2`, with `tee: tdx` and
`profiles`. Each profile is a complete tuple: `mr_td`, `rtmr_1`, `rtmr_2`,
`xfam`, `tdvfkernel`, and `tdvfkernelparams`, plus an identifier. Secure services
registers the tuple set under one RVPS reference, `coco_tdx_profiles_v2`.
The AS policy requires a complete tuple match. It must not construct an
approved configuration by choosing individual values from different profiles.
The SNP five-field schema remains unchanged and separate.

The service administrator installs the reviewed AS policy first and then
approved references. A reference-only update does not replace AS or KBS policy
and refuses a mismatched active CPU policy. The trusted system sends no AS
policy, signing key or private report to secure services in this handoff.

## Workload authorization and InitData

The public launch contract is a trusted-side drift check, not an attestation
claim. The hostile cluster may ignore every installer/launcher check. Key
authorization is enforced independently by guest agent policy and KBS.

The v2 workload authorization binds the release's CPU type and GPU mode to
its three exact resource paths and measured InitData. It includes the canonical
`init_data_sha256` and the expected hardware `init_data_claim`. SNP uses the
32-byte SHA-256 digest; TDX uses that digest followed by 16 zero bytes in the
48-byte MRCONFIGID. Nonzero TDX padding is invalid, not another accepted digest.

The generator and secure-services installer independently derive the expected
release-policy shape. CPU-only releases require exactly `cpu0`; GPU releases
require exactly `cpu0` and `gpu0`. Signed CPU evidence must identify the approved
TEE. Present failing GPU evidence cannot be ignored or downgraded to CPU-only.
Expected EAR vectors are exact: SNP `(3,2,3)`, TDX `(3,2,2)`, NVIDIA GPU `(3,2,3)`
for executables/hardware/configuration; the other five fields are zero.

Changing image digest, command, application security context, agent policy or
trust configuration requires a newly generated and separately approved release.
Image signing, encrypted layers, denied interactive access, denied host-visible
logs and strict resource-path checks remain unchanged across runtime targets.
Do not broaden CNI, guest policy or appraisal rules to pass a failed test.

## NVFlare peer proofs

CoCoAuthorizer obtains the signed EAR through the guest-local token API. It
selects SNP or TDX from authenticated evidence, verifies every required vector,
and produces a site/audience-bound proof using the guest-held attested key.
An ordinary server verifies the proof locally with its pinned AS public key;
it does not need to run in CoCo or contact Trustee to verify it.

Optional typed constraints include `cpu_tee`, `tdx_mr_td` and `tdx_rtmr_0`
through `tdx_rtmr_3`. The legacy `measurement` field remains SNP-only.
`init_data` pins the canonical digest for either TEE, with strict TDX padding
normalization. These are additional verifier restrictions, not substitutes for
KBS release policy. See [CCManager configuration](provision/CCMANAGER.md).

## Acceptance evidence

Retain successful and failing results privately with the code/runtime/policy
revisions. At minimum test a valid encrypted launch, fresh repeat evidence,
wrong challenge, tampered quote or boot log, unapproved/mixed reference tuple,
wrong CPU type, missing/failed GPU when required, changed InitData, wrong
resource path, altered image/command/security context, replayed or expired peer
proof, and RVPS persistence after restart. Confirm denials in the guest or
secure services, not solely the untrusted launcher's preflight.

Application logs remain confidential and must not be enabled for convenience.
Use authenticated application success, a verifier outside CoCo, and trusted
secure-services evidence for acceptance. Evidence collection alone is not an
encrypted-workload test; offline tests alone are not hardware validation.
