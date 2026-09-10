# Trusted platform-reference preparation

This procedure covers trusted artifact preparation, verified report collection,
and five-value JSON export. Run it only on the trusted platform system.

For the tested workload-specific trusted_system procedure, use
[SEC-SYS-LAUNCH-PROFILE.md](SEC-SYS-LAUNCH-PROFILE.md). It adds an approved
resource/artifact profile, captures the running QEMU command, and repeats a
fresh confidential launch. It exports only the five-value JSON; setting
`PLATFORM_REFERENCE_SKIP_SIGNING=1` skips signing-key generation in stage 06.
The signed bundle builders have been removed from this JSON-only kit.

## Stage 06: mechanical preparation

After stages 03, 04 and 05 succeed, display, hash, and run:

```bash
cd /home/operator/coco_deployment
sed -n '1,320p' trusted_system/06-prepare-platform-reference.sh
sha256sum trusted_system/06-prepare-platform-reference.sh
./trusted_system/06-prepare-platform-reference.sh \
  /home/operator/private-platform-reference/platform-reference.env
```

Stage 06 checksum-verifies the reviewed `sev-snp-measure` 0.0.13 wheel,
installs it in an isolated virtual environment, derives the firmware/kernel/
rootfs paths from the stage-03 Kata configuration, and creates
`platform-derived.env` and `platform-approval.env` beneath the private profile.
With `PLATFORM_REFERENCE_SKIP_SIGNING=1` (the supplied template), no signing
key or authority-key directory is created or needed.

## Stage 07: launch the fail-closed in-cluster rehearsal

Users do not create or manage a separate SEV-SNP VM. The trusted system must
already have Kubernetes and the approved `kata-qemu-nvidia-gpu-snp`
RuntimeClass installed. Stage 07 creates a random 64-byte challenge, builds a
checksum-pinned collector, publishes it by immutable digest to a short-lived TLS
registry on the trusted host, and launches it as a short-lived Pod with that
RuntimeClass. The registry's two-day private CA is embedded in measured
init-data so confidential guest pull remains TLS authenticated:

```bash
cd /home/operator/coco_deployment
sed -n '1,420p' trusted_system/07-run-snp-rehearsal.sh
sha256sum trusted_system/07-run-snp-rehearsal.sh \
  trusted_system/rehearsal-collector/Dockerfile \
  trusted_system/rehearsal-collector/collect-snp-evidence.sh
./trusted_system/07-run-snp-rehearsal.sh \
  /home/operator/private-platform-reference/platform-reference.env \
  /home/operator/coco_deployment/platform-reference-work/\
kata-3.29.0-nvidia-gpu-snp-genoa-v1/platform-approval.env
```

The Pod retrieves a fresh SNP report inside the Kata VM, fetches its complete
VCEK chain from AMD KDS, and emits one framed evidence archive through its
noninteractive log. The trusted host rejects missing or extra framing, unsafe
archive paths, checksum failures, a changed challenge, or a signed
`REPORT_DATA` field that does not equal the original challenge. It then calls
the internal TCB-recording helper automatically. It also records the report's
signed launch measurement and creates `rehearsal-evidence.txt`; stage 09 later requires that measurement
to equal the value extracted from the reverified signed report.
The namespace, Pod, and temporary registry are deleted on exit; evidence and
the collector image ID/digest remain beneath the private profile.

No `kubectl exec`, `kubectl cp`, SSH, listener, separately managed VM, or
operator-selected report value is used. Kubernetes logs are only the transport:
the fresh challenge and AMD signature authenticate the evidence.

## Internal helper: automatically record the reported-TCB floors

Under this deployment's assumption that the trusted rehearsal system always
meets the AMD security baseline, its verified `REPORTED_TCB` values become the
minimum release-policy floors. Stage 07 invokes this automatically. The direct
command below is retained for offline evidence imported through another
authenticated mechanism:

```bash
cd /home/operator/coco_deployment
sed -n '1,320p' trusted_system/record-reported-tcb.sh
sha256sum trusted_system/record-reported-tcb.sh
./trusted_system/record-reported-tcb.sh \
  /home/operator/private-platform-reference/platform-reference.env \
  /home/operator/coco_deployment/platform-reference-work/\
kata-3.29.0-nvidia-gpu-snp-genoa-v1/platform-approval.env \
  /path/to/attestation-report.bin \
  /path/to/amd-certificate-directory
```

The script fails before modifying the approval file unless `snpguest` verifies
the AMD certificate chain, the complete signed report, and its reported-TCB
field. It parses the signed binary `REPORTED_TCB` field according to the AMD
SEV-SNP ABI, atomically records all four decimal values in
`platform-approval.env`, sets `TCB_EVIDENCE_FILE`, and retains the report,
display output, certificates, and checksums under `reported-tcb-evidence/`.

This automates approval only because of the stated trusted-system baseline
assumption. Never run it against evidence supplied by the adversarial target
CoCo owner.

## Remaining evidence and approval

Stage 07 and its internal helper create both evidence files beneath the profile
directory:

- `REHEARSAL_EVIDENCE_FILE`: the collector image/base identities, pinned
  `snpguest` identity, RuntimeClass, Pod/challenge/report hashes, and the signed
  SNP launch measurement.
- `TCB_EVIDENCE_FILE`: created by the internal helper from the cryptographically verified
  reference-system SNP report.

Stage 07 records the signed report measurement in `rehearsal-evidence.txt`.
Stage 09 treats that value as authoritative only after it independently
rechecks the retained checksums, AMD certificate chain, report signature,
64-byte challenge binding, report hash, challenge hash, and four reported-TCB
bytes. Stage 09 also automatically obtains
the vCPU count from the pinned Kata TOML and the complete QEMU `-append` string
from `kata-runtime kata-env`. It first requires every parsed setting in the
installed SNP TOML to equal the pinned copy; comments and whitespace may differ.
It derives the vCPU CPUID signature from the
trusted AMD host and explicitly uses the standard SNP guest-features value
`0x1`. These launch inputs are not entered manually.

The remaining fields still require review. An operator or script running on an
adversarial cluster must never select any of these values.

## Stage 09: fail-closed finalization

After completing the approval file and stage 08 repeat rehearsal, display, hash, and run:

```bash
cd /home/operator/coco_deployment
sed -n '1,360p' trusted_system/09-finalize-platform-reference.sh
sha256sum trusted_system/09-finalize-platform-reference.sh
./trusted_system/09-finalize-platform-reference.sh \
  /home/operator/private-platform-reference/platform-reference.env \
  /home/operator/coco_deployment/platform-reference-work/\
kata-3.29.0-nvidia-gpu-snp-genoa-v1/platform-approval.env
```

Stage 09 rejects missing evidence, blank approval fields, malformed values,
ambiguous vCPU identity, any failed AMD signature or challenge check, any
evidence-hash mismatch, and any difference between the approval record and the
signed report. It creates:

```text
$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE/platform-reference.final.env
```

Stage 09 extracts the authoritative approved launch measurement from bytes
`0x90..0xbf` of the cryptographically verified stage-07 SNP report. It also
calculates an offline modeled measurement from the pinned artifacts and launch
inputs, but records that value explicitly as diagnostic only. A model/report
difference is preserved for investigation and never causes the model to replace
the hardware-signed value. Use only the final environment with stage 10 to
export the JSON; bundle builders are no longer part of this kit.

## Stage 10: export only five values for secure services

Run `10-export-platform-reference-values.sh FINAL_ENV OUTPUT.json` after
stage 09. It exports exactly the approved SNP launch measurement and four
minimum reported-TCB integers. Follow
[the five-value handoff procedure](../service/PLATFORM-REFERENCE-VALUES-HANDOFF.md)
for the concrete paths, authenticated transfer, secure services installation commands,
their order, and live readback/persistence verification.

The secure services owner trusts these contents. Evidence and artifacts remain on trusted_system;
no signing key or signed service archive is required. Stage 10 optionally emits
the separate admin launch contract. CoCo receives only the public chart and kit
digest pins for runtime setup, through provisioning_node. No signed CoCo bundle is built
or verified in this workflow. Receiving the JSON does not install it; the
secure-services administrator explicitly approves, installs and reads it back.
