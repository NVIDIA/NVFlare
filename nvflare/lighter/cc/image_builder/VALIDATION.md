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
