# CVM Builder validation

CVM Builder was imported into NVFlare from source commit
`712c070a8a0a40c5183194580b3625e3b950d982`. The source, test fixtures, guest
services and operator guides are maintained together with the provisioning adapter
in this repository. Repository formatting and license headers change source
fingerprints; build and approve new generic CVMs for this source snapshot.

## Shutdown warning fix — 2026-09-21 UTC

The shutdown warnings from the preceding CPU run are fixed. Both supervisors
now stop after application services, Docker and containerd. Their PID 1
failure/success power-off actions remain in place. The distro finalrd service
prepares its RAM-backed shutdown image at boot, before attestation, and the
measured hook includes the dynamically loaded `libmount` and `libblkid`
libraries and their dependencies. Startup checks reject missing libraries.

Fresh Intel TDX and AMD SEV-SNP CPU-only builds passed end-to-end testing:

- Both guests logged successful stops for `cvm_app.service`, `docker.service`
  and `containerd.service` before their supervisor stopped, then reached
  `Power down`. Neither shutdown contained the previous device-mapper busy,
  mount parsing, unmount, swap cleanup or waiting-for-process warnings.
- Three-round CPU jobs passed before and after periodic re-attestation in
  **8.20 seconds** and **8.18 seconds**, verifying all returned values and fresh
  request nonces. Both final sums were **529,920**.
- Initial and periodic hardware attestation passed, with periodic gaps of
  **301.88 seconds** for TDX and **300.74 seconds** for SNP. All **24 live
  Trustee HTTPS tests passed** against unmodified upstream Trustee v0.22.0.
- **240 Linux unit/policy tests passed without skips**. The new isolated
  finalrd integration test and the source-distribution-to-wheel packaging test
  passed. Scoped Black, isort, flake8, shell syntax and whitespace checks passed.
- No test processes, mounts or listening ports remained. The pre-existing AMD
  GPU VM retained its devices throughout the CPU-only run.

Both finalized manifests recorded runtime source SHA-256
`8fd810711cae8210e6e4681b4b0cfb8440ea06c9e6190fb8309a76cdc6f879fd`.
Generic construction took **323.53 / 324.06 seconds** (TDX / SNP), and vault
construction took **294.65 / 296.01 seconds**. Launch to observed secure-admin
readiness or client registration took **149.33 / 150.91 seconds**.

This candidate-mode run used native CVMs and SHA-256-verified OCI archives.
GPU operation, registry transfer, production approval, Kubernetes CoCo
orchestration and destructive security-failure acceptance were not repeated.
Generic images must be rebuilt, remeasured and reapproved for this change.

The shutdown-image integration test runs on an Ubuntu host with finalrd and
initramfs-tools installed. It uses a disposable mount namespace and chroot;
it does not invoke a power-off command or change the host's `/run` or `/etc`:

```bash
sudo env CVM_SHUTDOWN_TESTS=1 python3 -m unittest discover \
  -s tests/integration_test/lighter/cc/image_builder -p test_shutdown.py -v
```

## CPU hardware end-to-end after hardening — 2026-09-20

Fresh builds from commit `30d393340edacecd2df053764b0739c9415ec8f9` passed the
functional end-to-end test with an Intel TDX server and an AMD SEV-SNP client,
both without GPU. The AMD GPU was occupied by an existing workload, which was
left running. Each guest used 4 vCPUs, 8 GiB RAM and kernel `7.0.0-31-generic`.
The run completed on September 20 local time (September 21 UTC).

- Both three-round CPU jobs completed successfully, in **10.23 seconds** before
  scheduled re-attestation and **10.21 seconds** afterward. Every round checked
  all 1,024 returned values and a fresh request nonce; the final sum was
  **529,920**. The client executor also checked that `/dev/nvidia0` was absent.
- Initial and periodic hardware attestation passed on both participants. The
  observed re-attestation gaps were **301.98 seconds** for TDX and **300.69
  seconds** for SNP.
- **24 live HTTPS tests passed** against unmodified upstream Trustee v0.22.0,
  12 per isolated backend, using the current shipped administrative ACLs.
- Both finalized manifests recorded the expected runtime source digest:
  `a2779fee424916e40ba7f71123575e2db4da96df92fa4a606f75937c625b846f`.

| Stage | Intel TDX server | AMD SNP CPU client |
|---|---:|---:|
| Generic image construction | 301.50 s | 292.06 s |
| Hardware finalization | 97.24 s | 81.94 s |
| Application vault and delivery | 292.97 s | 295.59 s |
| Delivery archive verification and unpack | 55.28 s | 61.74 s |
| Launch to accepted attestation | 124.95 s | 105.04 s |
| Launch to observed application readiness | 150.08 s | 153.09 s |

The shared application image built in **44.04 seconds** and exported in
**5.78 seconds**. The AMD delivery archive transferred in **19.45 seconds**,
separate from verification/unpack. Importing AMD hardware reference evidence and
repackaging the builder-side generic bundle took another **57.73 seconds**.
Readiness means the first observed secure-admin connection or client registration;
job timing runs from submission through observed completion. Stages overlap.

Both delivered launchers shut down their CVMs, and both guest consoles reached
`Power down`. However, shutdown was **not warning-free**: both guests reported
that `/dev/dm-0` was busy during final device-mapper teardown; AMD also reported
mount/swap cleanup errors from its shutdown environment. These observations
prompted the shutdown fix and fresh hardware verification recorded above. Host cleanup
checks found no remaining test processes, mounts or listening ports. The existing
AMD GPU VM retained its GPU assignment.

This was a candidate-mode native-CVM run using SHA-256-verified OCI archives and
the unmodified delivered launchers. It did not test registry publication/pull,
production approval, Kubernetes CoCo orchestration, GPU functionality, configured
NTS time sources, or destructive quarantine/reopen acceptance. No policy or
measurement was relaxed to obtain the successful job results.

## Architecture and security review fixes — 2026-09-20

This source change set implements the review items recorded as design decisions
D18–D21 (signed approvals and authenticated pulls, quarantine on periodic
failure, kernel lockdown and console hardening, keyring-only volume keys with a
deterministic KDF, SNP policy from references, QMP-based orderly shutdown with an
explicit QEMU device set, container and service confinement, CIDR and resolver
restrictions, NTS-only time sources, a measured `vault_prescan` switch, finalize
identity scrubbing, `kbs-client` provenance, token lifetime and bundle-scoped
resource roles, a root-only lock directory, `--diagnostics`, git-ignored
`inputs/`, the `nvflare-cvmctl` entry point and ownership-checked wrappers).

New operator inputs: `inputs/kbs_client_build.json` from `cvmctl provenance`,
an Ed25519 acceptance signing key pair, `approval.public_keys` in
`cvm_project.yml`, `approval_public_keys` in the administration configuration,
`--signing-key` for `admin approve`, and `--archive-sha256`/`--cosign-key` (or an
explicit `--allow-unverified`) for `cvmctl pull`. Existing `approval.json` files
are unsigned and no longer count as approval.

A follow-up review found and fixed four regressions:

- Docker-published ports now apply the same inbound CIDR restrictions as host
  input ports, for both IPv4 and IPv6.
- Duplicate launches acquire the vault lock before touching the QMP socket,
  preserving the running VM's shutdown channel when a second launch is rejected.
- Failed reopens unmount and close partial vault activations. The supervisor
  terminates timed-out reopen process groups, revokes GPU readiness, and confirms
  cleanup before retrying. Cleanup failure propagates to the power-off path.
- Every reopen attempt and retry delay is bounded by the remaining 15-minute
  authorization window. Late success cannot restart the workload.

Linux follow-up verification:

- Builder unit/policy suite: **239 passed, no skips**, with the pinned policy
  evaluator. Includes a real Unix socket/flock duplicate-launch check, a real
  subprocess-tree timeout, and controlled failure/deadline regressions.
- Real block-device integration: **5 passed** on kernel `7.0.0-30-generic` and
  cryptsetup `2.8.4`, including repeated partial-unlock cleanup, frozen headers,
  kernel-keyring activation, corruption detection and sidecar validation.
- Packet integration: **1 test with 4 scenarios passed**, covering IPv4/IPv6
  host and DNAT paths, allowed/disallowed sources and unrestricted defaults.
  Routing and nftables changes are confined to disposable network namespaces.
- NVFlare adapter, CLI, vault supervisor and provisioning-output tests:
  **144 passed**, including the Linux builder-suite wrapper.
- Scoped Black, isort and flake8 checks passed, including the four formatting
  issues found by the review; Python syntax and `git diff --check` passed.

The preceding review run also passed **12 live HTTPS tests against unmodified
Trustee v0.22.0**, bundle-scoped ACL checks, and the
source-distribution-to-wheel test. Those results are separate from guest boot
validation. The original macOS builder tally was 201 passed, 24 skipped and 4
platform-related failures out of 229; all four cases pass on Linux.

The Linux checks above alone do not validate guest boot or QMP power-off; the
fresh CPU hardware run and its shutdown warnings are recorded above. Real-guest
quarantine/reopen, lockdown compatibility with the pinned NVIDIA modules, chrony
NTS startup and finalize hygiene inside a construction VM remain unvalidated.
The lab-host block tests do not establish compatibility with every guest kernel.
Every measured root must be rebuilt, remeasured and reapproved; earlier hardware
records do not cover these changes.

## Recorded TDX firmware input

The September 19 hardware run used the same direct-boot TDVF recorded in the
earlier standalone builder's validation record. Its digest was rechecked against
the retained firmware on the TDX host on September 19:

| Input | Value |
|---|---|
| EDK II release | `edk2-stable202605` |
| Source commit | `b03a21a63e3bd001f52c527e5a57feddb53a690b` |
| Platform DSC | `OvmfPkg/IntelTdx/IntelTdxX64.dsc` |
| Architecture / target / toolchain tag | `X64` / `RELEASE` / `GCC` |
| Build defines | `BUILD_SHELL=FALSE`, `SECURE_BOOT_ENABLE=FALSE` |
| Input filename | `inputs/OVMF.inteltdx.fd` |
| SHA-256 | `fb85eb43785820cf6ad50f13788c1f9362c76fa2861de8e3d2c0f2a6fcf7537f` |

These pins identify the tested artifact, not a guarantee that another build
environment emits identical bytes. Record the digest of the actual firmware in
the reviewed profile/bundle. This measured direct-boot path relies on TDX launch
measurements and Trustee approval; a Secure Boot deployment requires a separately
validated signed kernel/shim chain. No host firmware was flashed for this run.

## Service parser diagnostic review fix — 2026-09-19

Malformed application service files now produce a fixed configuration diagnostic
instead of exposing parser source lines, section names or option names. Regression
coverage exercises missing headers, malformed lines, duplicate sections and
duplicate options through the vault CLI and checks that rejection occurs before
profile, key or Trustee operations.

**8 CLI tests and 196 Linux unit/policy tests passed, with no skips.** Scoped
repository style checks passed for the builder, unit tests and integration tests.
This follow-up changes the shared service parser's error handling; the hardware
run below tested the preceding `e7b96f385` source, not rebuilt images containing
this fix.

## Hardware end-to-end after review fixes — 2026-09-19

A candidate-mode run passed with the runtime and build sources from commit
`e7b96f3851895428186dcafd75d1bb182ba8a4df`: an Intel TDX server without GPU and
an AMD SEV-SNP client with an NVIDIA H800 PCIe. The generic images were built on
September 18; the run resumed on September 19 after lab access and GPU availability
returned. Both application vaults were provisioned, published as OCI artifacts,
pulled by immutable digest and started with their delivered launchers.

- Both three-round CUDA jobs completed successfully: **14.25 seconds** before
  scheduled re-attestation and **16.29 seconds** after both participants re-attested.
  Every round executed H800 CUDA kernels without a CPU fallback and verified all
  1,024 returned values and the fresh request nonce. The final sum was **529,920**.
- Periodic attestation passed with observed gaps of **301.47 seconds** for TDX and
  **301.64 seconds** for SNP/GPU. No guest bootstrap or attestation failure was
  recorded in the completed deployment.
- **24 native Trustee HTTPS tests passed**, 12 per backend. The role-boundary
  integration test now uses a valid preissued policy token so requests reach
  Trustee's ACL, while the signing client's new local role guard remains intact.
  The test confirms resource POST/PUT/DELETE return 401 and leave both the resource
  and policy unchanged. Scoped repository style checks passed.

| Stage | Intel TDX server | AMD SNP/H800 client |
|---|---:|---:|
| Generic image construction | 313.81 s | 372.03 s |
| Hardware finalization | 107.84 s | 122.86 s |
| Application vault and delivery | 307.58 s | 324.59 s |
| Delivery registry publication | 42.45 s | 46.06 s |
| Delivery pull and unpack | 54.17 s | 56.66 s |
| Launch to accepted attestation | 121.77 s | 125.87 s |
| Launch to observed application readiness | 193.73 s | 155.41 s |

The shared application image built in **38.29 seconds** and exported in
**3.77 seconds**. Timings exclude pauses for lab access and GPU availability;
stages can overlap. Readiness is the first observed secure-admin connection or
client registration, so the server value includes delay while diagnosing the
client's first registry pull.

That initial pull failed when the AMD host's shared temporary filesystem had only
2.34 GiB free for a roughly 10 GiB delivery. Setting `TMPDIR` to a private directory
on the workspace disk allowed the same digest to be pulled successfully. No shared
temporary files were deleted, and no runtime source, delivery or attestation policy
was changed for the retry.

The run used unchanged upstream Trustee v0.22.0, Intel standard collateral,
NVIDIA's remote NRAS verifier, isolated candidate profiles and a lab registry.
It did not qualify Kubernetes CoCo orchestration, production approval or the full
destructive acceptance suite. Both test CVMs and isolated backends were stopped
afterward, and the GPU was released.

## Resource-role and boot review follow-up — 2026-09-18

**195 Linux unit/policy tests passed with no skips**, including the pinned Regorus
engine and Linux memfd tests. New coverage verifies explicit resource-role signing,
unchanged default policy-role signing, safe diagnostics for forbidden preissued
tokens, malformed Docker/OCI archives, conventional signal exit codes, and reference
boot behavior on both TEE platforms. Signed administrative test tokens are verified
cryptographically; no live Trustee deployment was used for these role tests.

When a vault disk is already present, boot skips only the reference-only report
probe. The vault's fresh hardware binding check still precedes key authorization;
an absent disk still requires an all-zero hardware binding before emitting reference
frames. Scoped repository style checks and `git diff --check` passed. The hardware
end-to-end run below was not repeated for this follow-up.

## Launcher and diagnostics review fixes — 2026-09-18

- **186 Linux unit/policy tests passed with no skips**, using the pinned Regorus
  engine. Added coverage includes disabling `vmport` in every launch mode,
  interruption during process creation and state publication, forced child cleanup,
  orphaned file locks, configuration-error redaction, duplicate verity arguments,
  and short audit writes. Existing read-only NBD and mount options remain in use
  by the hardware, periodic-attestation and storage integration tests.
- **Nine real-QEMU process checks passed** with paused, non-confidential QEMU
  guests and disposable disks. SIGTERM, SIGHUP, SIGQUIT and SIGINT during both
  process creation and state publication stopped the child and removed its runtime
  record. After SIGKILL of the launcher, shutdown detected the surviving QEMU's
  actual byte-range disk lock and reported it without signalling an unverified
  process. All smoke-test QEMU processes were stopped and their locks released.
- Scoped repository style checks and `git diff --check` passed.

These checks validate the review fixes; they do not repeat the TDX/SNP/GPU
end-to-end run recorded below. The updated guest sources and initramfs hook
require fresh images, measurements and acceptance before production approval.

## CLI refactor and distribution packaging — 2026-09-18

A fresh candidate-mode end-to-end run passed with an Intel TDX server, an AMD
SEV-SNP client and an NVIDIA H800 PCIe. Both generic images were built from a
clean Ubuntu 26.04 cloud image, finalized on their target hardware, published as
OCI artifacts, combined with separately encrypted NVFlare application vaults,
pulled by digest and launched using the delivered host scripts. Secure admin
access and client registration succeeded.

- The three-round CUDA job completed in **16.29 seconds**. It executed GPU
  kernels without a CPU fallback and checked every returned value and request
  nonce; the final sum was **529,920**.
- Scheduled re-attestation passed on both participants, with observed gaps of
  **301.84 seconds** (TDX) and **301.80 seconds**
  (SNP/GPU). A second three-round CUDA job then passed in **16.28 seconds**.
- The backend used unchanged upstream Trustee v0.22.0, with Intel's standard
  collateral channel and NVIDIA's remote NRAS verifier. An initial test-harness
  configuration omitted the NRAS verifier and correctly failed closed; after
  correcting that configuration, the same images passed. No attestation policy
  or approved measurement was relaxed.
- **178 Linux unit/policy tests**, **116 adapter/supervisor tests**, **24 native
  Trustee HTTPS cases** (12 per backend), and **4 real storage tests** passed.
  The macOS Linux-only wrapper was skipped; its underlying suite ran on Linux.
- Release packaging now includes the complete builder source and assets,
  including the NVAT compatibility patch. The distribution regression test
  builds an sdist and a wheel from it, compares all builder files byte-for-byte,
  checks executable metadata, and runs the CLI and GPU input tests from the
  extracted wheel. This test and scoped repository style checks passed.

| Stage | Intel TDX server | AMD SNP/H800 client |
|---|---:|---:|
| Generic construction | 324.36 s | 358.91 s |
| Hardware finalization | 103.03 s | 122.85 s |
| Application vault and delivery | 309.66 s | 317.90 s |
| Delivery registry publication | 43.14 s | 46.45 s |
| Delivery pull and unpack | 55.40 s | 55.17 s |
| Launch to accepted attestation | 64.86 s | 124.21 s |
| Launch to observed application readiness | 74.70 s | 150.18 s |


The shared NVFlare application image built in **41.56 seconds** and exported in
**4.61 seconds**. Timings are observed wall-clock values; stages overlap. Registry
pull timings are from the fresh pulls; launch/readiness timings are from the
successful retry with the corrected Trustee configuration. Readiness is the
observed secure-admin connection or client registration time.

This run used isolated candidate profiles, an isolated test registry and native
Trustee with the delivered CVM launchers. It did not qualify Kubernetes CoCo
orchestration or the complete destructive hardware acceptance suite. The test
CVMs and isolated backends were stopped afterward, and the GPU was released.

## PR review verification — 2026-09-18

- **171 Linux unit/policy tests passed**, including the pinned Regorus engine,
  monotonic re-attestation scheduling with a simulated 240-second appraisal,
  operation-specific errors that suppress secret output, QGS configuration
  variants, repository key substitution, and NVAT provenance validation.
- A clean upstream Trustee v0.22.0 `restful-as` cryptographically verified its
  public signed SNP evidence fixture and issued an affirming ES256 EAR under the
  CVM CPU policy with isolated test references. The committed fixture tests its
  original signature and hex encodings in guest, AS and resource-policy checks.
  This uses archived public evidence; it is not a fresh hardware challenge or a
  live SNP vault-unlock test. See the [fixture provenance and regeneration steps](../../../../tests/unit_test/lighter/cc/image_builder/fixtures/README.md).
- The actual Intel lab host's headerless `/etc/qgs.conf` passed preflight parsing.
- NVAT commit `0c1be386a8fbb8f2766a6a556d10df86f5fed9d3` built successfully on
  Ubuntu 26.04 using only the recorded const-correctness patch. The resulting
  library version is 1.2.0, soname `libnvat.so.1`, using `libxml2.so.16`.
  The documented source/provenance checks passed. This build's library SHA-256
  was `7477e3d947910d3d2b1bfb628962c2c2bdc59b123c900c0933356ddd17379bb3`;
  record the hash produced by the selected build environment rather than treating
  this test result as a published binary artifact.
- **GPU Stage 1 construction from the clean Ubuntu cloud image passed**, including
  authenticated repository setup, pinned 580.178.04 driver packages, Container
  Toolkit 1.20.0-1, NVAT installation, hardening, verity root construction and OCI
  packaging. It used a plain construction VM with no GPU assignment and deferred
  hardware measurements; the resulting bundle remains unapproved.
- Scoped repository style checks passed for the builder, unit tests and
  integration helpers (`./runtest.sh --skip-install -s <directory>`).

These checks do not qualify a new live TDX/SNP/GPU job, GPU denial timing, or the
full hardware acceptance suite. The changed measured runtime and GPU inputs need
new generic images, reference collection and acceptance before production approval.

## CoCo v0.23.0 / Trustee v0.22.0 migration — 2026-09-17

The current implementation uses unmodified upstream Trustee commit
`512fed65642015b849f38fb13bfdec7806639987`. The previous custom Rust verifier,
attester and source patch are removed. The default generic profile advances to
`cpu-2026.09-r4`; earlier bundle approvals do not cover this migration.

The isolated Linux compatibility tests use the official upstream KBS image
`ghcr.io/confidential-containers/staged-images/kbs` at digest
`sha256:92c24e93f60fa259bab4f5404576d70a0edeb48a1b08e0c33ed159526fc87a9a`.
The earlier kbs-client image/digest claim was incorrect and is withdrawn; it is
not a retrievable client pin. CVM clients must follow the clean-source build in
[BUILD_GUIDE.md](BUILD_GUIDE.md), with the requested hardware features and the
resulting executable SHA-256 recorded. Sample-attester tests do not qualify those
hardware clients. The release checkout remained clean. The Rego evaluator uses upstream's Regorus
0.11.0 with its RVPS query extension contract.

- **Combined Linux builder, policy and live HTTPS suite: 152 passed, 20 skipped.**
  This includes 12 real HTTPS cases covering encrypted key retrieval, cross-vault
  denial, favorable/incomplete GPU EARs, forged transport signers, stale tokens,
  idempotent/conflicting uploads, revocation, administrative mutation denial,
  named RVPS queries, persisted policy verification, and the actual AS denying
  sample evidence under the installed CPU policy. Remaining skips require
  hardware or destructive storage/network-fault opt-in.
- **NVFlare provisioning/entrypoint checks: 33 passed, 1 skipped** on macOS.
  The skipped Linux contract wrapper is covered by the standalone Linux run.
- **Scoped repository style checks passed:** Black, isort, flake8 and agent-skill
  checks through `./runtest.sh -s nvflare/lighter/cc/image_builder`.

CPU/GPU Rego tests enforce reference expiry, fresh EARs, approved measurements,
TCB and driver/VBIOS references, exact GPU count and distinct identities. The
backend uses upstream ACLs and local_fs namespaces. Production templates also
require an installed resource policy and read-only AS/reference/resource mounts;
the disposable HTTP harness does not qualify production orchestration controls.

The HTTPS GPU cases use signed EAR fixtures. **No new physical TDX/SNP/GPU job or
full hardware acceptance is claimed for this upstream migration.** NVIDIA
cryptographic verification now follows CoCo's implementation; deleted custom
verifier tests are not evidence for upstream behavior. Rebuild, remeasure and
repeat hardware acceptance before approving the new profiles. All hardware and
custom-verifier results below are historical and apply to their recorded sources.

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

Run from the repository root on the Linux test host. Populate the configuration
templates and use isolated lab credentials, resources and ports.

```sh
export PYTHONPATH="$PWD/nvflare/lighter/cc/image_builder:$PWD/tests/unit_test/lighter/cc/image_builder${PYTHONPATH:+:$PYTHONPATH}"
cargo build --locked --release --manifest-path tests/unit_test/lighter/cc/image_builder/policy_engine/Cargo.toml
python3 -m unittest discover -s tests/unit_test/lighter/cc/image_builder -v
sudo env PYTHONPATH="$PYTHONPATH" CVM_STORAGE_TESTS=1 \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_storage.py -v
sudo env PYTHONPATH="$PYTHONPATH" CVM_NETWORK_TESTS=1 \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_firewall.py -v

env CVM_HTTP_TESTS=1 CVM_LAB_DIRECTORY=/path/to/isolated-lab \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_http.py -v

sudo env PYTHONPATH="$PYTHONPATH" CVM_HARDWARE_TESTS=1 CVM_EXTENDED_AGENT=1 \
  CVM_NETWORK_FAULTS=1 \
  CVM_BUNDLE=/path/to/test-bundle CVM_VAULT=/path/to/test-delivery \
  CVM_OTHER_RESOURCE=keys/TESTED_BUILD_ID/OTHER_EXISTING_BINDING \
  CVM_HARDWARE_OUTPUT=/path/to/new-evidence-directory \
  python3 -m unittest discover -s tests/integration_test/lighter/cc/image_builder -p test_hardware.py -v
```

Add `CVM_GPU_HARDWARE_TESTS=1` only on a configured NVIDIA CC GPU host; it
enables the mandatory NRAS failure/poweroff acceptance case for GPU profiles.

For the hardware fault tests, explicitly include `tests/integration_test/lighter/cc/image_builder/lab_guest_agent.py` as
executable application payload and `tests/integration_test/lighter/cc/image_builder/app_acceptance.service` as a service,
with ports 18080/18081 and the generic HTTP fixture. The agent refuses non-test
profiles and is never installed by Stage 1. `CVM_OTHER_RESOURCE` must identify an
existing different key under the same tested bundle. Tests mutate independent
vault file copies and keep serial logs and result hashes. The interrupted-load
fixture accepts the deliberately stopped container's termination status while
preparing its fault; integrity supervision stays active and reboot restores the
normal measured application exit policy.

`tests/integration_test/lighter/cc/image_builder/boot_http.py` separately checks arbitrary generic HTTP test images without
the fault-injection payload. It launches a disposable delivery copy and preserves
the boot log, exact bundle digest and result.

`tests/integration_test/lighter/cc/image_builder/periodic_hardware.py` coordinates the two periodic-denial checks: wait for
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
building `tests/unit_test/lighter/cc/image_builder/policy_engine`:

```sh
python3 tests/run_nras_policy_test.py /path/to/patched-trustee \
  --policy-eval "$PWD/tests/unit_test/lighter/cc/image_builder/policy_engine/target/release/cvm-policy-eval"
```

Both standard verifier tests and this additional verifier-to-policy test passed
on Linux. The Linux builder suite again passed 136 tests with 31 opt-in skips;
all 10 isolated live HTTPS tests passed against the rebuilt KBS. Clean-checkout
patch reproducibility, scoped project style and Python license checks also passed.
This is a claim-schema and signature/policy test, not a physical GPU attestation
test.

## 2026-09-18 — native CoCo resource administration

CVM Builder now uses the existing CoCo Trustee deployment directly. Removed the
three Trustee systemd templates and the Python key-administration server.
Project configuration uses `trustee.url`, `trustee.ca`, and a pre-issued
`trustee.admin_token_file`; issuer signing keys stay with the deployment operator.
Native resource POST/DELETE replaces the custom create-only/revocation API.

Validation performed against unmodified Trustee v0.22.0:

- Linux builder contracts with the pinned Rego evaluator: **140 passed**.
- Linux NVFlare wrapper and CLI checks: **5 passed**.
- NVFlare adapter and CLI checks on macOS: **96 passed, 1 skipped**
  (the Linux-only wrapper; exercised separately above).
- Isolated live HTTPS tests: **12 passed**. Covered native upload and replacement,
  deletion and recreation, forged resource tokens, resource/policy role separation,
  encrypted key retrieval, default-policy selection, cross-vault denial, expired
  and wrong-policy EARs, and composite CPU/GPU authorization using signed fixtures.
- Scoped project style checks passed. The disposable Trustee process was stopped
  after testing and its listening port released.

The overwrite/recreation tests document native CoCo behavior. This revision does
not provide the removed adapter's immutable key publication or permanent
revocation tombstones. CoCo operators must fence active uploads and preserve
current deletions and bundle retirement policy during backup recovery. Existing
adapter deployments require the migration steps in `TRUSTEE_GUIDE.md`.

Signed EAR fixtures are not physical attestation evidence. No VM or GPU workload
was started, and the paused two-machine end-to-end test remains paused.


## 2026-09-18 — separate CVM Python packages

Replaced the mixed `builder` package with `cvm.build`, `cvm.runtime`, `cvm.host`,
`cvm.trustee`, `cvm.artifacts` and `cvm.common`. Build configuration, physical-host
access, guest hardware access, policy generation and artifact verification now
have separate owners. Public shell entry points and configuration fields remain
unchanged. Python operator scripts are thin entry points into the packages.

Validation:

- Linux standalone unit suite with the pinned Trustee Rego evaluator: **145 passed**.
- Linux NVFlare wrapper and public CLI entry points: **13 passed**.
- NVFlare adapter and CLI tests on macOS: **104 passed, 1 skipped** (Linux wrapper).
- Isolated-interpreter checks passed for both staged payloads and for the runtime
  installed by the construction provisioner. Tests enforce package dependency
  boundaries, include lazy imports, and cover the source fingerprint of runtime,
  provisioning and payload-manifest changes.
- Normal NVFlare lighter pytest collection: **264 tests collected**, with the
  standalone tests kept behind the Linux subprocess wrapper.
- Opt-in integration suite imports successfully: **32 skipped** because hardware,
  storage and live HTTPS tests were not enabled for this refactor.
- Scoped project style checks and explicit checks of all moved Python files passed.

No CVM was built or booted and no GPU workload was started. The installed guest
paths and contents changed, so rebuilt generic images, fresh measurements and new
hardware acceptance/approval are required before production use. Existing
self-contained deliveries are unaffected; new builds should use newly approved
generic bundles. The paused two-machine end-to-end test remains paused.

## 2026-09-18 — three guest systemd units

Consolidated the guest lifecycle into `cvm_bootstrap.service`,
`cvm_integrity.service` and `cvm_app.service`. Bootstrap emits reference frames,
checks the firewall and clock, opens and configures the vault, then sends
readiness before synchronously starting application services. Its supervisor
runs isolated periodic children every five minutes or on SIGUSR1. PID 1 now
handles security shutdown directly through unit failure/success actions.

Stage 1 installs measured bootstrap rules in `/etc/nftables.conf` and enables
the distro nftables unit. Docker socket activation is masked; its measured
vendor unit opens the Unix socket directly instead of requiring `docker.socket`.
Development images use the same three unit files without dependency rewriting.

Validation:

- Linux standalone unit suite with the pinned Trustee Rego evaluator: **155 passed**.
  Covers gate ordering, wrong-binding denial before KBS/monitor startup, monitor
  readiness before scan, dev behavior, generated unit dependencies, actual notify
  datagrams, a real queued SIGUSR1, periodic success/failure, and GPU-revocation
  failure. Provisioning tests check the three-unit layout and measured firewall.
- Linux NVFlare wrapper and public CLI entry points: **13 passed**.
- NVFlare adapter and CLI tests on macOS: **104 passed, 1 skipped** (Linux wrapper).
- `systemd-analyze verify` passed for all three units and the modified Docker
  vendor unit on systemd 259.5. Offline `systemctl --root` checks confirmed that
  Docker/containerd remain disabled, Docker's socket remains masked after disable,
  and chrony, nftables and bootstrap are enabled.
- Bootstrap and application nftables rules passed syntax checks and successive
  application in a private network namespace. The host firewall was not changed.
- Opt-in integration suite imports successfully: **32 skipped**. Hardware helpers
  now request periodic checks via SIGUSR1, inspect the atomic tick record, and
  assert the three-unit inventory, masked Docker socket and enabled firewall.
- Scoped project style and explicit Python formatting/lint checks passed.

No generic CVM was built or booted and no GPU workload was started. The paused
hardware end-to-end test remains paused. Rebuilt SNP/TDX images still require
reference collection, positive boots, wrong-binding denial, monitor death and
corruption, periodic/GPU denial timing, development boot, and crash-consistency
acceptance before approval. Earlier boot and timing evidence does not validate
this changed shutdown path.
