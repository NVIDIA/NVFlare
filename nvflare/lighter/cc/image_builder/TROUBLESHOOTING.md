# CVM Builder troubleshooting

Start with the first failing lifecycle stage. Re-running `cvmctl vault` after an
uncertain key upload can replace a native KBS resource, so preserve
`build_failure.json` and `provisioning.json` before retrying. Use
`--diagnostics PRIVATE_DIRECTORY` to retain stderr only for commands documented
as not handling secrets; never publish that directory without review.

## Build or preflight fails

- Run `./cvmctl preflight host --platform intel_tdx --firmware ...` or
  `--platform amd_sev_snp` explicitly. The default is TDX.
- For TDX, verify `kvm_intel.tdx=1`, `nohibernate`, active QGS on vsock port
  4050, collateral configuration, platform registration, and the site quote
  probe. Follow [TDX_TROUBLESHOOTING.md](TDX_TROUBLESHOOTING.md) for QGS cache
  recovery and `standard` versus `early` TCB channels.
- For SNP, verify the firmware path, `sev_snp` CPU flag,
  `kvm_amd.sev_snp=Y`, `/dev/sev`, and IOMMU groups. A
  `SNP_LAUNCH_START` error can mean the approved guest policy is unsupported on
  that host; do not weaken the policy without a new profile and review.
- `finalize` rejects TDX and SNP evidence whose reported TCB or configuration
  differs from `approved-tcb-references.json`. Inspect and approve the new
  values, increment `profile_version`, and rebuild rather than accepting a CVM
  that Trustee will deny at every boot.

## Trustee cannot attest or release a key

- Check that the guest's measured `kbs_url`, KBS CA, AS public key, token issuer,
  selector, and policy ID match the running deployment. An acceptance service
  with disposable PKI cannot be replaced after the bundle is built.
- Confirm the selector is mapped to exactly the installed immutable CPU/GPU AS
  policies and that its profile-specific RVPS record exists and is unexpired.
- A resource upload denial usually means the token is expired, has the wrong
  issuer/audience or role, or the endpoint ACL does not cover the exact
  `keys/<build_id>/<binding_id>` path. Mint a short-lived scoped token with
  `cvmctl admin token`; do not copy the issuer private key to the builder.
- For SNP KDS outages, configure `snp_verifier.vcek_sources` with
  `OfflineStore`, install the exact host/TCB VCEK under the upstream path, and
  rerun the offline-collateral acceptance case. A certificate for another host
  or firmware TCB is not a fallback.

## `cvmctl vault` fails

- Create the workload archive with `docker save --output application.tar`.
  Gzipped `.tar.gz` streams and OCI archives are not Docker save archives.
- Ensure `cvm_image` names the exact profile directory (for the checked-in
  template, `cvm_cpu-2026.09-r5`) or an immutable registry digest.
- On a capacity error, include retained archive bytes, expanded layer chains,
  application files, filesystem/Docker overhead, and expected runtime writes.
  Increase `vault_drive_size`; do not disable the preflight.
- After an uncertain native resource upload, have the Trustee operator reconcile
  or revoke that resource before another build attempt.

## The CVM starts and powers off

Review the private serial log and KBS logs together. Early poweroff is the
fail-closed result for invalid measurement/TCB references, stale or wrong policy
selection, missing offline collateral, failed time synchronization, vault
integrity errors, or unavailable GPU appraisal. Do not turn off the watchdog or
integrity monitor to keep the workload running. Verify the delivery and generic
artifact signatures/digests again, then reproduce with the exact manifest's
acceptance workload if the cause remains unclear.

For evaluation without TEE or KBS dependencies, use the
[30-minute `--dev` walkthrough](README.md#30-minute-dev-mode-walkthrough). Dev
artifacts cannot be promoted or used as acceptance evidence.
