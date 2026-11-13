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

The CLI reports a safe KBS cause for native upload failures: ``KBS connection
failure`` means check endpoint reachability/DNS/listener and firewalls; ``KBS TLS
verification failure`` means check the CA, hostname, certificate validity and
local clock; ``KBS HTTP NNN`` means the server returned that status. For 401/403,
check the scoped administration token and endpoint ACL. No request body, token,
key, backend response body or raw exception details are printed.
These uploads run in Python, so ``--diagnostics`` does not create a subprocess
stderr capture for them. An unacknowledged upload can still have reached KBS;
retain ``build_failure.json`` and reconcile it before retrying.

## The CVM starts and powers off

Review `console-<timestamp>.log` beside the materialized delivery and the KBS
logs together. The launcher prints the current serial-log path and retains the
three newest files with mode 0600, including failures before `/applog` mounts.
Each new capture retains the first 16 MiB. Reaching that cap or encountering a
write failure stops file capture with a warning; terminal output continues.
If capture cannot be created it warns, so retain the attached terminal output.
Raw guest syslog is not exported. For missing job/client logs, use the
[authenticated CC log-retrieval workflow](../../../../docs/user_guide/confidential_computing/operations.rst#retrieve-job-logs-and-early-boot-diagnostics).
Early poweroff is the
fail-closed result for invalid measurement/TCB references, stale or wrong policy
selection, missing offline collateral, failed time synchronization, vault
integrity errors, or unavailable GPU appraisal. Do not turn off the watchdog or
integrity monitor to keep the workload running. Verify the delivery and generic
artifact signatures/digests again, then reproduce with the exact manifest's
acceptance workload if the cause remains unclear.

The bootstrap writes `CVM bootstrap failed [phase=...]` to stderr before
asking PID 1 to enforce watchdog failure. The write is nonblocking: a full
or unavailable console does not postpone shutdown, and raw exception
details remain private.

Root-disk corruption **after** startup is different from the initramfs marker:
the kernel reports ``dm-verity device corrupted`` and panics/powers off. There
may be no ``CVM_STARTUP_FAILURE=verity`` or Python ``phase=`` line. Retain the
serial output, stop the delivery, and re-authenticate/rematerialize the approved
generic image. Do not disable verity or interpret the absence of a marker as
an attestation or application fault.

### Cold boot waits for `time_sync`

A sealed production guest starts without chrony's drift history. Its first
NTS samples can correct the offset before they span enough time to estimate
frequency uncertainty (skew). The initial `iburst` is therefore not evidence that
the clock gate should already pass. The normal polling interval can be about
64 seconds; boot readiness can vary across otherwise identical launches.

The cold-boot gate allows 180 one-second `chronyc waitsync` tries after an
initial, best-effort 30-second correction wait. It requires correction at most
0.5 seconds and skew at most 1000 ppm before attestation. Runtime `chronyc burst`
is not used to force convergence. Periodic checks keep their shorter budget.

If the final gate fails, verify the measured NTS server names, DNS resolution,
outbound UDP 123 and TCP 4460, and time-server TLS reachability. Use the serial
phase identifier and network-side diagnostics; the production guest has no
interactive shell. A small offset alone does not establish a synchronized
clock. Keep both quality limits and the fail-closed startup gate in place.

For evaluation without TEE or KBS dependencies, use the
[30-minute `--dev` walkthrough](README.md#30-minute-dev-mode-walkthrough). Dev
artifacts cannot be promoted or used as acceptance evidence.
