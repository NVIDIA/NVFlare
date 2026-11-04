# CVM Builder production acceptance

Production approval is specific to a finalized CVM manifest and target
hardware/software configuration. Use this page with the [build
workflow](BUILD_GUIDE.md#normal-build-seal-and-run-flow) and [Trustee
guide](TRUSTEE_GUIDE.md). Repository checks or a successful application launch
cannot replace the required acceptance and signed approval.

## Supplied profile

The checked-in `config/cvm_profile.yml` is a **candidate** with
`production_ready: false`. Its manifest cannot receive production approval.
The profile pins Ubuntu 26.04, kernel `7.0.0-31-generic`, and its complete
package and firmware inputs. A kernel version alone does not establish that a
profile is suitable for production.

Prepare a separately versioned production profile with reviewed inputs and
complete acceptance on each target platform. Do not alter a finalized manifest,
reuse another bundle's report, or relabel the supplied candidate. The
`production_ready` declaration is a prerequisite for approval, not evidence
that the profile has passed acceptance.

CPU-only and GPU profiles have different acceptance requirements. The portable
SNP profile uses `snp_single_socket: false`; enabling the optional single-socket
restriction requires a separate approved profile supported by every target
host's firmware, kernel and QEMU stack.

## Production approval

1. Select reviewed OS, package, firmware, CPU/GPU and Trustee inputs. Keep the
   production KBS URL, transport CA and AS signing key consistent with the
   measured profile. An acceptance endpoint must present that same pinned
   identity; unrelated disposable lab PKI is insufficient for production.
2. Construct and finalize the exact generic bundle. Finalization checks approved
   TCB and configuration references for the selected CPU TEE. Preserve the
   manifest, source fingerprints, launch shape, package pins and private
   approval inputs.
3. Prepare an isolated acceptance endpoint with the reviewed policies and
   references. Qualify its selector behavior and administrative authorization,
   install the candidate, and seal a candidate vault for the exact manifest.
   Follow the build guide's split workflow so all prerequisites exist before
   invoking the acceptance runner.
4. Collect every applicable acceptance result on the target platform. Retain
   signed `result.json` records and private diagnostics. Bind results to the
   exact `cvm_manifest.json` digest and CPU platform.
5. Aggregate the records with `cvmctl acceptance-report`, then have the acceptance
   authority sign approval with `cvmctl admin approve`. The evidence signer and
   approval signer are trusted administrative roles, not runtime-host credentials.
6. Install the approved bundle and publish authenticated generic artifacts.
   Seal production application vaults without `--candidate`. Protect private
   receipts, scoped resource credentials and acceptance evidence separately
   from the public OCI delivery.

The exact check names and profile selection are defined by
[`required_acceptance_checks()`](cvm/artifacts/bundle.py). The
[aggregator](cvm/artifacts/acceptance.py) verifies signatures and rejects missing,
failed, duplicate, unknown or inapplicable checks, as well as records for another
manifest or platform. The resulting approval contains evidence digests; it is
not an unsigned local pass flag.

## Required acceptance areas

Every production profile must cover these areas:

| Area | Required check names |
|---|---|
| Measured boot and generic workload | `boot_measurements`, `generic_container`, `initramfs_no_kbs`, `root_overlay_capacity` |
| Binding and resource isolation | `local_binding`, `wrong_binding`, `header_snapshot`, `cross_vault_key_denial`, `exclusive_attachment` |
| Authenticated storage and interruption | `payload_corruption`, `reboot_after_writes`, `interrupted_journal`, `interrupted_docker_load`, `integrity_monitor_failure`, `root_disk_corruption` |
| Appraisal, policy and administration | `negative_appraisal`, `policy_selection`, `unauthorized_administration`, `policy_readback`, `durable_key_retry`, `key_revocation`, `rollback_retirement` |
| Clear sidecars and guest hardening | `clear_sidecar_scan`, `read_only_input_disks`, `writable_applog`, `ssh_service_and_socket_disabled`, `clock_synchronized_before_attestation` |

Additional checks depend on the profile:

| Profile | Additional required check names |
|---|---|
| CPU-only | `attestation_quarantine_recovery` |
| AMD SEV-SNP, with or without a GPU | `snp_collateral_availability` |
| NVIDIA confidential GPU | `gpu_negative_key_denial`, `gpu_positive_key_release`, `gpu_policy_selection`, `cross_class_denial`, `periodic_gpu_denial` |

CPU-only periodic appraisal failures use bounded quarantine recovery. GPU
Ready-state revocation is terminal for the running VM: stop the workload,
revoke readiness, close the vault and power off. Do not substitute CPU recovery
for the GPU denial requirement.

Acceptance must exercise failures at the enforcement boundary. A denied
application request is not proof that Trustee withheld the key; a powered-off
VM is not by itself proof of orderly workload containment. Check the expected
key-release result, guest state transition and workload behavior together.

## Record the deployment boundary

Keep a private, reviewed record of:

- Exact manifest and OCI digests, profile version, source fingerprints and
  signatures, including the selected CVM and application artifacts.
- Server model, CPU/GPU identity and topology, host kernel/QEMU, firmware,
  microcode, guest pins, driver/VBIOS and CC-mode settings.
- Trustee revision, immutable policy IDs/selectors, reference-value namespaces,
  reference expiry, service identity and scoped resource authorization.
- Network allowlists, time sources, storage policy, application inputs and the
  approved recipients of outputs and logs.
- Signed acceptance evidence, approval authority and recovery/retirement state.

A change to measured software, launch shape, policy or approved hardware inputs
requires review of the affected profile and fresh acceptance. A new application
vault may reuse an approved compatible generic CVM; it still needs fresh keys,
bindings and the deployment-specific application review.

## Recorded TDX firmware input

Obtain the TDVF artifact from the trusted build or platform authority. Record
its source revision, build flags, toolchain and SHA-256 digest before using it
as a profile input. Keep the firmware and resulting launch measurements bound
to the same approval. Do not copy a digest or TCB floor from another machine's
report as a substitute for an approved input.

The [TDX troubleshooting guide](TDX_TROUBLESHOOTING.md) explains quote generation,
QGS cache recovery and the `standard`/`early` collateral selection used by the
verifier. Neither channel selection nor successful quote generation supplies
missing platform firmware mitigations.

## Reproduce acceptance

Follow the [exact-manifest acceptance flow](BUILD_GUIDE.md) and select the
applicable cases from the configured site runner. Use isolated resources and
candidate vault copies; failure injection must not target a running production
workload or another operator's VM/GPU. Preserve private evidence until the
approval authority has reviewed it, then revoke temporary candidate resources.

For an environment without TEE or Trustee dependencies, use the
[`--dev` walkthrough](README.md#30-minute-dev-mode-walkthrough). Development
artifacts cannot be promoted or used as production acceptance evidence.
