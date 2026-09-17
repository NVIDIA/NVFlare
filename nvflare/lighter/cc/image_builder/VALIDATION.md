# CVM Builder validation

CVM Builder was imported into NVFlare from source commit
`712c070a8a0a40c5183194580b3625e3b950d982`. The source, test fixtures, guest
services and operator guides are maintained together with the provisioning adapter
in this repository. Repository formatting and license headers change source
fingerprints; build and approve new generic CVMs for this source snapshot.

## Recorded hardware validation

The September 16, 2026 candidate-mode run used NVFlare commit
`2beb0212fe1563f6abc9bd355fef42353abee146` and builder commit
`298d822b9aab1f2fcfdcf739e498390576ea9c2b`, before the combined `2.9` PR.
It ran an Intel TDX server and an AMD SEV-SNP client with an NVIDIA H800 GPU.
Both generic CVMs, encrypted application vaults and complete OCI deliveries were
built, published, retrieved and launched. Signed startup kits, mutual TLS,
CPU/GPU appraisal, vault unlock and client registration succeeded.

The adapter used shared `cvm_project.yml` settings, local-folder and registry
inputs, generated deployment IDs and Docker image identity derived from the
archive. Per-build configuration omitted `key_service`, `deployment_id` and
`platforms`. The default participant output directories were used, and OCI archive
and manifest digests were verified.

| Test | Result |
|---|---|
| Three-round CUDA computation | Completed in 14.58 s; GPU output and fresh nonce verified |
| GPU linear regression, three rounds of 20 local steps | Completed in 14.18 s; loss fell from 2.32943 to 0.000339 |
| Periodic attestation on both participants | Passed after approximately five minutes |
| CUDA computation after periodic attestation | Completed in 12.57 s |
| Provisioning adapter and supervisor tests | 118 passed |
| Builder GPU/project-configuration tests | 11 passed |
| Isolated HTTPS key-service checks | 8 passed |

The CUDA computation had no CPU fallback. The regression job used GPU kernels
for gradients and loss, with CPU reductions. Both VMs and the isolated backend
were stopped after testing, and the GPU was restored to its prior host driver.

| Stage | Intel TDX | AMD SEV-SNP + H800 |
|---|---:|---:|
| Generic construction and OCI packaging | 282.80 s | 439.85 s |
| Hardware finalization and repackaging | 102.18 s | 116.95 s |
| Vault and complete-delivery build | 300.35 s | 308.12 s |
| Complete-delivery publication | 38.05 s | 37.21 s |
| Pull, verification and extraction | 49.43 s | 54.58 s |
| Boot to observed application readiness | 68.98 s | approximately 91.13 s |

The shared application image took 37.47 s to build and 3.48 s to save. Generic
construction ran concurrently; application vaults were provisioned sequentially.
Cached Ubuntu inputs, Trustee binaries and Docker base layers were reused.
These are functional lab timings, not production performance benchmarks.
The run did not produce production approval receipts. Hardware testing has not
been repeated for the combined `2.9` source snapshot.

## Repository checks

Validation of the combined `2.9` source import passed:

- Linux: 279 provisioning/integration tests, including the standalone builder contract suite.
- macOS: 278 tests passed; the Linux-only contract runner was skipped.
- Pinned policy evaluator: five resource-policy and two appraisal tests passed.
- Black, isort, flake8, agent-skill lint and Python license-header checks passed.
- Thirteen shell entrypoints passed syntax checks; YAML, example paths and Markdown links were checked.
- All 23 imported runtime Python modules retain the source commit's parsed syntax trees; changes to those modules are license headers only.

From the NVFlare repository root, run:

```sh
python3 -m pytest tests/unit_test/lighter/cvm_builder_test.py \
  tests/unit_test/lighter/vault_adapter_test.py \
  tests/unit_test/lighter/vault_supervisor_test.py \
  tests/unit_test/tool/provision_output_test.py -q
```

The regular NVFlare unit-test suite runs the builder's unprivileged contracts in a
separate Python process on Linux. CLI help checks also run on macOS. The builder
contracts require Linux `memfd` and `/proc`; storage, live HTTPS and hardware
acceptance remain explicit opt-in tests below. Policy tests run when the pinned
Rust policy evaluator has been built or `CVM_POLICY_EVAL` points to that binary.

## Reproducing tests

Run from the builder directory on the Linux test host. Populate the configuration
templates and use isolated lab credentials, resources and ports.

```sh
cargo build --locked --release --manifest-path tests/policy_engine/Cargo.toml
sudo env CVM_STORAGE_TESTS=1 \
  CVM_POLICY_EVAL="$PWD/tests/policy_engine/target/release/cvm-policy-eval" \
  python3 -m unittest discover -s tests -v

env CVM_HTTP_TESTS=1 CVM_LAB_DIRECTORY=/path/to/isolated-lab \
  python3 -m unittest discover -s tests -p test_http.py -v

sudo env CVM_HARDWARE_TESTS=1 CVM_EXTENDED_AGENT=1 \
  CVM_NETWORK_FAULTS=1 \
  CVM_BUNDLE=/path/to/test-bundle CVM_VAULT=/path/to/test-delivery \
  CVM_OTHER_RESOURCE=keys/TESTED_BUILD_ID/OTHER_EXISTING_BINDING \
  CVM_HARDWARE_OUTPUT=/path/to/new-evidence-directory \
  python3 -m unittest discover -s tests -p test_hardware.py -v
```

Add `CVM_GPU_HARDWARE_TESTS=1` only on a configured NVIDIA CC GPU host; it
enables the mandatory NRAS failure/poweroff acceptance case for GPU profiles.

For the hardware fault tests, explicitly include `tests/lab_guest_agent.py` as
executable application payload and `tests/app_acceptance.service` as a service,
with ports 18080/18081 and the generic HTTP fixture. The agent refuses non-test
profiles and is never installed by Stage 1. `CVM_OTHER_RESOURCE` must identify an
existing different key under the same tested bundle. Tests mutate independent
vault file copies and keep serial logs and result hashes. The interrupted-load
fixture accepts the deliberately stopped container's termination status while
preparing its fault; integrity supervision stays active and reboot restores the
normal measured application exit policy.

`tests/boot_http.py` separately checks arbitrary generic HTTP test images without
the fault-injection payload. It launches a disposable delivery copy and preserves
the boot log, exact bundle digest and result.

`tests/periodic_hardware.py` coordinates the two periodic-denial checks: wait for
its `ready.json`, complete and verify the isolated backend change, then create
`proceed` in that evidence directory. Restore reference values before subsequent
tests. A revoked identity must be replaced with a newly built vault.

Validation runs should publish results, manifests, source fingerprints, archive
hashes, serial logs and image artifacts through the applicable CI or test-job
artifact store. Do not commit generated evidence, credentials or key resources to
the source repository. After validation, stop all task VMs, NBD/crypt mappings,
temporary transfer servers, SSH tunnels and isolated Trustee services, remove
temporary credential copies, and restore host settings.

## Remaining production gates

The current GPU candidate has exercised driver/toolkit installation, H800
passthrough, NVIDIA remote appraisal, and the positive NVFlare GPU workload path
recorded above, including computation after periodic reattestation. The complete
GPU failure-matrix acceptance remains pending; no GPU production profile is
approved. TDX-vTPM and in-place
vault resize/re-encryption remain outside this implementation. The tested vault
size establishes functional behavior, not production capacity or throughput.

No production `approval.json` is emitted by these test runs. A production operator
must select supported pins, endorse the reference values, validate the complete
deployment-specific acceptance matrix and approve each exact generic bundle.
Sector/snapshot replay and continued use of an already released key remain the
documented limitations; key deletion prevents future retrieval.

## 2026-09-17 — PR boundary fixes and composite GPU authorization

This change set replaces the post-unlock GPU check with a single composite RCAR
transaction. CPU-only resource rules retain their previous bytes. GPU rules
require the exact configured count of distinct NVIDIA EAR submods, with matching
policy IDs, affirming status and trust vectors. The guest collects evidence;
Trustee verifies the NRAS overall/device signatures, digest linkage, issuer,
freshness and challenge binding, then applies the generated GPU policy and RVPS
driver/VBIOS approvals before releasing the key.

The compatibility revisions remain Trustee
`a2570329cc33daf9ca16370a1948b5379bb17fbe` and guest-components
`591d0bb45cd7a2c66f3778428940c40f7eec3b7d`. The reviewed composite boundary patch
SHA-256 is `94de3a62f7abc62664be7734667bb300fefdc6aec52ae99e4662c03678bc720e`.
The default profile is now `cpu-2026.09-r2`; use a new GPU profile and rebuild,
remeasure and reapprove affected bundles. Existing hardware results and approvals
do not cover these new sources.

Validation performed:

- Linux standalone suite with the pinned Rego evaluator: **136 passed, 31
  skipped** (167 discovered). Skips cover opt-in hardware/storage/live-backend
  tests and the separately executed clean-checkout provenance test.
- Isolated live KBS/key-service HTTPS tests: **10 passed**. Includes real encrypted
  resource responses, CPU-only release, valid composite GPU EARs, missing/extra/
  duplicate/wrong-policy/non-NVIDIA GPU submods, expired tokens, forged
  builder/admin/server/untrusted signers, create/revoke/retry behavior, byte-exact
  policy readback, POST 403 and PUT/DELETE 405 with no key/policy mutation.
- Backend Rust tests: **2 passed**, including valid ES384 NRAS detached EATs and
  invalid device/overall signatures, digests, nonces, issuers, timestamps, claims
  versions and device counts, plus unsupported/sample TEE error dispatch.
- KBS and the composite SNP/TDX/NVIDIA client both build with `cargo build
  --locked --release`. No dependency version update is required.
- Clean-checkout provenance test: **1 passed**, applying the patch twice to the
  exact pins, checking the configured digest, and detecting modified companion
  guest sources.
- NVFlare provisioning/entrypoint tests on macOS: **96 passed, 1 skipped** (the
  Linux-only contract wrapper). The standalone Linux suite above supplies that
  platform coverage.
- `./runtest.sh -s nvflare/lighter/cc/image_builder` passed Black, isort, flake8
  and agent-skill checks. Archive tests also verify that raw reference reports
  and plaintext content fingerprints are excluded and that a delivered GPU
  policy can run using only its packaged Python helpers.

The HTTPS and NRAS tests use disposable signed fixtures; they are **not physical
GPU attestation evidence**. The occupied AMD GPU workload was left running.
No new two-machine federated GPU job or GPU hardware acceptance is claimed.
`CONFORMANCE.md` C1 and the GPU negative-release row remain **Evidence pending**.

Before production approval, run `gpu_negative_key_denial` with CC disabled,
tampered/replayed evidence and too few GPUs; `gpu_positive_key_release`;
`gpu_policy_selection`; `cross_class_denial`; and `periodic_gpu_denial`. Confirm
no key/allow record/vault mapper before a successful GPU decision and measure
fail-closed timing. The initial transaction budget is 240 seconds for GPU
profiles (60 seconds CPU-only), with a 300-second periodic service deadline.
The hardware harness's NRAS fault-injection cases require
`CVM_GPU_HARDWARE_TESTS=1` and `CVM_BACKEND_LOCAL=1`, on a dedicated lab setup with
KBS on the same host: the network fault must affect the **backend's** NRAS egress.
Its backend-outage check is additional coverage, not a substitute for the
CC-disabled/tampered/replayed/missing-device production acceptance cases.

## 2026-09-17 — NRAS detached-claim compatibility follow-up

A second PR review exposed an assumption in the verifier fixtures: they repeated
`x-nvidia-ver` and `x-nvidia-device-type` in the detached GPU token. NVIDIA's
[documented NRAS example](https://docs.nvidia.com/attestation/quick-start-guide/latest/attestation-examples/hopper_single_gpu.html#decoded-nras-token)
places the version in the signed overall token and identifies the detached token
through the signed `GPU-0` digest. The corrected fixture reproduced the original
`Unexpected NVIDIA claims version` rejection before the fix.

The verifier now checks the version on the overall token, preserves every signed
detached claim, and derives the policy's version/device marker from the verified
overall token and its GPU digest linkage. Conflicting detached markers still
fail. Required driver/VBIOS, secure-boot, RIM and OCSP claims are not defaulted.
The strict policy intentionally rejects older illustrative responses that omit
its required OCSP freshness or response-validity claims.

The updated boundary patch digest is
`13a32cdb2ac6e3dc6be9738961378bff9c41c32b7729ddb9df3cc9d1eef5ac66`.
The default profile/policy advance to `cpu-2026.09-r3` / `cvm-cpu-r3`; GPU
examples use `gpu-2026.09-r3` / `cvm-gpu-r3`. Rebuild, remeasure and reapprove
affected bundles. Physical GPU acceptance remains pending.

The reproducible regression signs an independent v3 fixture with disposable
ES384 keys, verifies it using the shipped Rust verifier, and passes the resulting
claims to the generated policy using the pinned Rego engine. It checks a positive
decision and denial for missing secure-boot, driver/VBIOS, RIM, report-signature
and OCSP-freshness claims. Run it after applying the current Trustee patch and
building `tests/policy_engine`:

```sh
python3 tests/run_nras_policy_test.py /path/to/patched-trustee \
  --policy-eval "$PWD/tests/policy_engine/target/release/cvm-policy-eval"
```

Both standard verifier tests and this additional verifier-to-policy test passed
on Linux. The Linux builder suite again passed 136 tests with 31 opt-in skips;
all 10 isolated live HTTPS tests passed against the rebuilt KBS. Clean-checkout
patch reproducibility, scoped project style and Python license checks also passed.
This is a claim-schema and signature/policy test, not a physical GPU attestation
test.
