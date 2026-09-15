# NVFlare token-API runtime profile

Protected NVFlare clients need the guest-local `/aa/token` API, in addition to
the resource API used for encrypted images. This package derives a reviewed
configuration with `agent.guest_components_rest_api=all` in
`hypervisor.qemu.kernel_params`. The upstream Kata image and chart remain
digest-pinned. The customization is explicit; it is not an unmodified upstream
configuration and does not reuse its old launch measurement.

## Trusted system

Use a new `PLATFORM_PROFILE` and follow the existing [stages 03–10](trusted_system/SEC-SYS-LAUNCH-PROFILE.md).
Stage 03 preserves the extracted upstream configuration and artifact hashes. It
creates two additional private profile files:

- `approved-kata-config.toml`: deterministic derivation with only the REST API option changed.
- `kata-runtime-profile.json`: derivation version, capability, and upstream/approved configuration hashes.

The shared `kata-runtime-profile.py` helper preserves other parameters, including
repeated `pci=` options. Missing parameters are appended; an existing single
`resource` or `attestation` value is changed to `all`. Duplicate or unknown values
and unsupported TOML layouts fail closed. Rerunning host enablement is idempotent.

Stage 04 installs the derived configuration after Kata rollout, before profile
capture. Stage 05 requires the option; stage 06 uses the approved configuration.
Stage 09 verifies original artifact hashes, re-derives the expected configuration,
checks provenance, compares the installed file, and checks the actual QEMU launch
for exactly one required option and the approved configuration hash. It retains
all existing report-signature, nonce, artifact, TCB and repeat-rehearsal checks.

Do not edit only the installed TOML, weaken stage 09, or add a per-Pod kernel
override. Do not merely re-finalize old evidence: repeat both rehearsals under
the new configuration. Keep independently approved TCB floors; do not lower them
to make the changed profile pass. Other known rehearsal failures must still be
resolved without bypassing those checks.

After finalization, stage 10 emits the same five-value JSON for secure services.
The separate admin contract now uses `coco-approved-workload-launch/v2` and carries
`guest_token_api: guest-local-aa-token/v1`, verified against the hash-bound profile
and captured command line. This is a coordination requirement, not a new signed
attestation claim or authorization policy.

## Secure services and provisioning node

The provisioning node coordinates delivery as before. The secure-services owner
reviews and installs the newly collected measurement using the existing
[reference installer](service/PLATFORM-REFERENCE-VALUES-HANDOFF.md). Do not silently
authorize both old and new measurements; choose the reviewed measurement set.
CPU/GPU appraisal and workload-specific resource-release rules remain required.

The provisioning node authenticates and installs the new admin contract and its
SHA-256 pin, then regenerates the workload release and handoffs. Stage 30 rejects
v1 contracts or contracts missing the required capability. The generator still
does not allow arbitrary kernel-parameter annotations. Existing releases and
delivered handoffs are not modified or revoked automatically.

## CoCo IT

Use an updated assembled CoCo kit. The normal stage 30 bootstrap and stage 35
re-pinning apply the same reviewed option after Kata deployment. Stage 60 checks
both the configuration file and the effective parameters reported by `kata-env`.
The normal host configuration is:

```text
/opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml
```

For a read-only configuration check from the CoCo kit:

```bash
sudo python3 lib/kata-runtime-profile.py check \
  /opt/kata/share/defaults/kata-containers/configuration-qemu-nvidia-gpu-snp.toml \
  --runtime /opt/kata/bin/kata-runtime
```

Kata replacement/reinstallation can overwrite local configuration. Run the
updated re-pinning procedure and verification again before launching new workloads.
Existing guest VMs are not updated by editing host configuration; launch a new
Pod with the regenerated handoff. CoCo host checks are operational diagnostics,
not trusted evidence: secure services still verifies the guest measurement.

## Acceptance checks

Run an actual packaged protected NVFlare client, not only the demo workload.
Verify guest-local proof generation, ordinary-server signature/peer verification,
successful registration, and encrypted workload resource release. Verify that a
configuration without the required API cannot be approved/exported and that an
unapproved launch is denied by secure services.

Never expose the API through a host port or Service. Do not log/dump its raw
response: it includes a private TEE key. Test through `CoCoAuthorizer.generate()`
and report success/failure, not the key material. The separate clock-skew/retry
and CCManager error-return issues are not fixed by enabling this route.

Offline configuration and packaging tests do not establish hardware E2E success.
