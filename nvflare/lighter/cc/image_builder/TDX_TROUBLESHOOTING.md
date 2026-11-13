# Troubleshoot Intel TDX attestation

Use this guide to diagnose Intel QGS/DCAP quote generation, collateral
selection and Trustee appraisal. Follow the host prerequisites in
[BUILD_GUIDE.md](BUILD_GUIDE.md#1-prepare-the-build-host) and backend setup
in [TRUSTEE_GUIDE.md](TRUSTEE_GUIDE.md#5-configure-upstream-kbs).

## Identify the failing stage

The path is: TDREPORT in the guest, signed quote from host QGS, quote verification
in Trustee, CPU appraisal against approved references, then vault-key release.
A working TDREPORT or an active `qgsd` service does not establish that the later
stages work.

| Evidence | Investigate | Next step |
| --- | --- | --- |
| No quote; cache expiry followed by PCS HTTP 400, `0xb011` and `0x11001` | QGS/PCK cache recovery | Follow [QGS cache expiry](#qgs-cache-expiry). |
| PCS HTTP 401 or 403 | PCS credentials or authorization | Validate the configured credential privately; restarting QGS does not fix authorization. |
| PCS HTTP 400 with a real nonzero encrypted PPID, or HTTP 404 for PCK certificates | Registration, request fields, TCB or service configuration | Check platform registration and the selected PCS/PCCS configuration; do not repeatedly restart QGS. |
| PCS HTTP 5xx, DNS, TLS or connection errors | Collateral service, network, proxy or time | Restore connectivity and correct clock/certificate configuration. |
| Quote produced and verified, but `tcb_status` is `OutOfDate` | Verifier collateral channel and platform firmware | Follow [TCB channel selection](#tcb-channel-selection). |
| Favorable quote status, but CPU appraisal or key release is denied | Claim names, approved measurements, reference expiry, resource binding or policy | Check the signed appraisal and installed policy/reference versions. |

First confirm the basics: `kvm_intel.tdx=1`, `nohibernate`, reviewed TDVF, and
the intended QGS transport. The supplied CVM profile uses vsock CID 2, port 4050;
`/etc/qgs.conf` must explicitly set `port = 4050`. A commented port selects a
Unix socket. TDX initialization can be lazy, so a missing early boot message
alone is not evidence that TDX is disabled.

## QGS cache expiry

### Recognize the failure

With direct Intel PCS through QCNL, quote generation can fail after a
cached PCK response expires. Look for PCS HTTP 400 together with provider
error `0xb011` and TDX quote error `0x11001`. Confirm the complete failure
sequence before applying the recovery below.

The DCAP 1.27 cached quote path intentionally passes no encrypted PPID during
quote sizing/generation. QCNL represents that missing value as 384 zero bytes
and excludes the field from its cache key, so a cached lookup can succeed. After
cache expiry, it can send the placeholder to PCS and receive HTTP 400.
This does not imply that the platform's actual PPID is zero. See the pinned
[TDX quote path](https://github.com/intel/confidential-computing.tee.dcap/blob/DCAP_1.27/QuoteGeneration/quote_wrapper/tdx_quote/td_ql_logic.cpp#L1679-L1692)
and [QCNL request construction](https://github.com/intel/confidential-computing.tee.dcap/blob/DCAP_1.27/QuoteGeneration/qcnl/certification_service.cpp#L199-L220).

QGS keeps quoting contexts per worker thread. This provider failure does not
trigger its reinitialization path, which retries for an attestation key that
has not been initialized. Restarting QGS discards the cached contexts so
initialization can obtain a real encrypted PPID and refresh the PCK response.
See [QGS context initialization and retry handling](https://github.com/intel/confidential-computing.tee.dcap/blob/DCAP_1.27/QuoteGeneration/quote_wrapper/qgs/qgs_ql_logic.cpp#L113-L212).

### Recover and verify

1. Confirm the combined cache-expiry/HTTP-400/provider-error sequence. An error
   code alone does not identify this failure. Preserve sanitized timestamps,
   component versions and result codes before restarting.
2. Coordinate a brief interruption to quote generation, including guests with
   periodic attestation deadlines. Restart the host QGS service:

   ```sh
   sudo systemctl restart qgsd.service
   sudo systemctl is-active qgsd.service
   sudo systemctl show qgsd.service \
     -p ActiveEnterTimestamp -p MainPID -p NRestarts -p ExecMainStatus
   ```

3. Boot a bounded test TD and request a signed quote bound to a new random
   challenge. Verify the expected report-data binding in the TDREPORT and quote,
   and verify the quote with current collateral. A standalone diagnostic may
   use a fresh 64-byte challenge; an application probe must retain its attester
   protocol's prescribed challenge/public-key binding.

   From the builder directory, a site-provided probe can be invoked through:

   ```sh
   sudo ./cvmctl preflight host --firmware inputs/OVMF.inteltdx.fd \
     --quote-probe /usr/local/sbin/site_tdx_quote_probe
   ```

   `site_tdx_quote_probe` is an executable supplied by the deployment operator;
   it is not shipped by CVM Builder. It must check actual signed quote generation
   and verification, then shut down its test TD.

4. Check high-level QGS events. This filter emits only recognized event text,
   rather than complete debug lines that could contain request details:

   ```sh
   sudo journalctl -u qgsd.service --since '-10 minutes' --no-pager --output=cat \
     | rg --only-matching \
       -e 'Cache (expired|missed)' \
       -e 'HTTP status code: [0-9]{3}' \
       -e 'Updated file cache successfully|Failed to get quote config' \
       -e 'tee_att_(init_quote|get_quote_size|get_quote) return (0x[0-9a-fA-F]+|[Ss]uccess)' \
       -e '0xb011|0x11001'
   ```

   Event availability depends on the configured logging level. A missing info
   message alone is not a failed probe. Keep any detailed diagnostic logs
   restricted; do not publish PCS subscription keys, encrypted PPIDs, complete
   request URLs, TDREPORTs or raw quotes. If inspecting the encrypted PPID,
   record only whether it has the expected length and is nonzero.

5. Require successful quote initialization, sizing and generation, valid
   challenge binding, and a favorable verifier result with unexpired collateral.
   For the direct-PCS cache-refresh path, confirm HTTP 200 and a refreshed PCK
   cache entry. Then test the intended Trustee appraisal and vault-key retrieval.
   Shut down the diagnostic TD and check that its QEMU process exited.

Do not delete all QGS caches, alter cache timestamps, disable TLS verification,
or bypass appraisal. A QGS restart recovers quote generation; it does not change
the firmware baseline used by Trustee.

### Monitor for recurrence

A QGS restart refreshes quoting contexts; it is not a firmware update or a
permanent fix for the underlying cache-refresh path. Monitor the actual
cache lifetime and the combined failure sequence. Use a fresh,
challenge-bound quote probe for health checks. If the failure recurs,
retain sanitized diagnostics for an Intel DCAP/QGS issue report and
review a maintained PCCS or provisioned-cache deployment.

## TCB channel selection

### Why an unchanged host can start reporting OutOfDate

Host QCNL and the Trustee verifier have separate collateral settings.
Upstream Trustee v0.22.0 defaults to `early` and fetches collateral itself. Changing only the
host's `/etc/sgx_default_qcnl.conf` therefore does not configure this Trustee
verifier. The default is visible in
[Trustee's pinned DCAP configuration](https://github.com/confidential-containers/trustee/blob/512fed65642015b849f38fb13bfdec7806639987/deps/verifier/src/intel_dcap/mod.rs#L25-L51).

Intel's `standard` channel allows a mitigation deployment grace period; `early`
applies newer TCB recovery requirements. A platform can pass one baseline while
being out of date under the other. Select the channel as a deployment security
policy, following [Intel's TCB recovery guidance](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/best-practices/trusted-computing-base-recovery.html).

A successful verification API call means the operation completed; inspect
the quote result and collateral-expiration status separately to determine
whether the platform satisfies the selected TCB baseline.

### Configure the intended baseline explicitly

The checked-in [KBS configuration](trustee/kbs.json) selects `standard`. Merge
this object into the selected standalone or CoCo KBS configuration, preserving
its other AS settings and any `nvidia_verifier` configuration:

```json
{
  "attestation_service": {
    "verifier_config": {
      "dcap_verifier": {
        "collateral_service": "https://api.trustedservices.intel.com/sgx/certification/v4/",
        "use_secure_cert": true,
        "tcb_update_type": "standard"
      }
    }
  }
}
```

Apply the change through the selected deployment's configuration and rollout
process. Verify that the running KBS instance loaded the intended configuration,
then request a fresh quote through that backend. Record the selected channel
alongside the deployed policy and reference approvals.

For deployments requiring the early baseline, set `tcb_update_type` to `early`
and update the platform firmware to meet that baseline. Check the vendor's
supported BIOS, microcode, authenticated-code-module and TDX component versions
and required reboot sequence. Restarting QGS cannot supply missing firmware
mitigations, and passing `standard` does not establish compliance with `early`.

### Check the policy claim and complete the application test

The raw Intel DCAP result `OK` is exposed by Trustee v0.22.0 as the claim
`input.tdx.tcb_status == "UpToDate"`. The
[CPU policy](config/attestation_policy.rego) requires `UpToDate`, unexpired
collateral, approved measurements/TCB, and acceptable configuration. It continues
to reject `OutOfDate`; do not add a permissive fallback to make a test pass.
Inspect `tcb_status_current` as well when the verifier exposes it.

Changing policy artifacts or guest runtime content requires the normal bundle
versioning and approval workflow. Keep the installed AS policy bytes, bundle
policy digest and deployment receipt consistent; a matching policy name alone
does not establish this.

Validate recovery through the complete path:

1. Generate and verify a fresh, correctly bound TDX quote with the intended
   collateral channel.
2. Confirm favorable CPU appraisal and exact approved measurements/references.
3. Confirm vault-key release, successful vault unlock and application startup.
4. Check a real application connection, then periodic attestation when qualifying
   a production deployment.
