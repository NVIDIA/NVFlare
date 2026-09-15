# Trusted-system kit: five-value JSON handoff

For standalone transfer, use the [assembled role kit](../README.md#assemble-self-contained-role-kits),
not this source directory alone. The assembled kit includes its shared dependencies.

Start with [../CONFIGURATION.md](../CONFIGURATION.md). No private bootstrap
configuration, source workload or approved measurements are included.

This kit prepares exactly five approved platform-reference values for secure services
from fresh, verified SNP evidence on the trusted-system machine. It no longer
contains signed platform-reference or public CoCo installation bundle builders.

Stage 10 can also export a separate small approved workload launch contract
for provisioning_node. Both outputs go through provisioning_node; only the five-value JSON
goes to secure services. CoCo receives public chart/pins, not a signed runtime bundle.
This contract is not a runtime installation bundle. Follow
[the admin handoff guide](../admin/APPROVED-LAUNCH-PROFILE.md).

Follow [SEC-SYS-LAUNCH-PROFILE.md](SEC-SYS-LAUNCH-PROFILE.md) for the complete
tested commands, prerequisite installation, launch-profile capture and
verification limits. Use `sec-sys-launch-profile.env.example` with
`PLATFORM_REFERENCE_SKIP_SIGNING=1`; no signing key is needed.

## Execution order

1. Fresh-machine setup: `01-install-rehearsal-host-tools.sh "$CONFIG"`, then
   `02-install-kubernetes.sh`.
2. Runtime acquisition and installation: `03-fetch-kata-artifacts.sh`, then
   `04-install-rehearsal-runtime.sh`.
3. Approved profile and preparation: `05-define-approved-launch-profile.py`,
   then `06-prepare-platform-reference.sh`. Fill the four independently reviewed
   TCB minimums in `platform-approval.env` before collecting evidence.
4. Verified collection and repeat: `07-run-snp-rehearsal.sh`, then
   `08-repeat-profile-rehearsal.py`.
5. Finalization and export: `09-finalize-platform-reference.sh`, then
   `10-export-platform-reference-values.sh`.

The numeric prefixes now match execution order. Stage 08 is
a separate repeat check, not automatically invoked by the finalizer. Stage 09
requires and revalidates its evidence for an approved workload profile.
The ten entry-point scripts above must remain together with their supporting
files.

## Optional GPU availability check

After a successful rehearsal, run the following from the package root:

```bash
python3 trusted_system/verify-gpu-in-pod.py "$PROFILE"
```

See
[GPU-VERIFICATION.md](GPU-VERIFICATION.md). This diagnostic is not an additional
numbered JSON-export stage and does not modify the approved values.

## Internal dependencies: keep, but do not run separately

- `bootstrap/lib/common.sh` and `bootstrap/10-install-kubernetes.sh`: entry
  points into shared source, materialized when assembling a standalone kit.
- `bootstrap/config.env`: private Kubernetes configuration; the common kubeadm
  template is in `../shared/bootstrap/templates/` in source and is materialized
  as `bootstrap/templates/kubeadm.yaml.in` in assembled kits.
- `bootstrap/config.env.example`: fallback/bootstrap configuration template.
- `rehearsal-collector/Dockerfile` and
  `rehearsal-collector/collect-snp-evidence.sh`: collector image and guest code.
- `capture-running-launch.py`: captures the actual Pod-associated QEMU launch.
- `record-reported-tcb.sh`: verifies and retains reported TCB evidence, checks
  it against independently approved minimums, and never rewrites those minimums.
- `export-workload-launch-profile.py`: stage-10 helper for the separate admin
  contract, derived from stage-09 hash-bound launch evidence.

Keep the trusted platform environment and source workload YAML as described
in the procedure. Stage 06 installs the measurement binary; stage 09 calls it
directly for diagnostic output, without a standalone recalculation script.

## Handoff and retained evidence

Transfer only `platform-reference-values.json` through an authenticated
channel. Secure services receive no private keys, signatures, artifact archives, reports
or policies in this handoff. Follow
[the five-value handoff procedure](../service/PLATFORM-REFERENCE-VALUES-HANDOFF.md).

Retain launch profiles, reports, certificates and verification records on the
trusted side. The report's verified measurement is authoritative; the offline
model is diagnostic only. Workload identity and GPU attestation remain separate
release-policy checks.

To check a complete local package after transfer, run from the package root:

```bash
python3 validate-package.py --assembled
```

Use Python 3.11+ and Bash as described in the root README. This checks files
present, script syntax, links and shared dependencies, not regression behavior,
package completeness, byte-for-byte integrity, source authenticity or hardware
attestation. In a source Git checkout, omit `--assembled`. Authenticate the source
and transfer channel independently.
