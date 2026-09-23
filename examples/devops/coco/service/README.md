# Service administrator kit

For standalone transfer, use the [assembled role kit](../README.md#assemble-self-contained-role-kits),
not this source directory alone. The assembled kit includes its shared dependencies.

First configure `platform.env` from `platform.env.example` as described in
[../CONFIGURATION.md](../CONFIGURATION.md). No certificates or credentials
are included.

Start with [SERVICE-INSTALLATION.md](SERVICE-INSTALLATION.md). It describes
installation on a fresh Ubuntu 24.04 secure services host, including the commands run by
the service administrator, trusted coordinator, provisioning-node owner and
CoCo IT.

The resulting secure services are:

- Trustee KBS behind Nginx TLS on **8443**;
- Attestation Service with CPU/GPU verifiers;
- RVPS with persistent LocalFs references; and
- a private OCI registry behind Nginx TLS on **5000**.

All backend listeners are loopback-only. This kit uses the pinned post-v0.21
Trustee source and 128 KiB HTTP request-head patch, not Trustee v0.21.0.

## Installation order

Script prefixes now match the fresh-install execution order: **01 through 11**.
Run as `service_operator` from `/home/service_operator/coco-service-admin`, stopping on any
failure. Before step 02, receive and approve the five-value JSON file and set:

```bash
VALUES=/home/service_operator/incoming-platform/platform-reference-values.json
python3 ./lib/platform-reference-values.py validate "$VALUES"
```

| Order | Command |
|---|---|
| 01 | `bash ./01-install-host-tools.sh` |
| 02 | `bash ./02-install-platform-reference-values.sh "$VALUES" --approve-platform-reference-values --configure-only` |
| 03 | `bash ./03-preflight.sh` |
| 04 | `bash ./04-build-trustee-main.sh` |
| 05 | `bash ./05-deploy-trustee.sh` |
| 06 | `bash ./06-configure-trustee-tls.sh` |
| 07 | `bash ./07-harden-kbs-admin-audience.sh` |
| 08 | `bash ./08-deploy-private-registry.sh` |
| 09 | `bash ./09-install-platform-policy.sh --approve-pinned-snp-platform` |
| 10 | `bash ./10-verify-platform-reference-values.sh "$VALUES"` |
| 11 | `bash ./11-verify-service.sh` |

Before step 03, configure the registry publisher's current egress IPv4/32 as
described in the full installation guide. After step 11, perform that guide's
RVPS restart/readback test and distribute only the appropriate public
certificates and publisher credentials to their intended recipients.
Do not execute every shell file with a wildcard: steps 12/13 need a separate
workload handoff. Host-specific teardown scripts are not included.

### Rerunning stage 05 resets workload authorization

`05-deploy-trustee.sh` is a deployment/bootstrap operation, not a policy-preserving
service restart. **Every run overwrites
`$TRUSTEE_ROOT/kbs/data/kbs-policy/resource-policy.rego` with the kit's
default-deny policy**, discarding all previously approved workload release rules.
The overwrite happens before Compose starts, so a later deployment failure does
not restore the old policy. Stage 05 does not make a recovery backup for you.

After KBS loads/reloads this policy, new requests for protected resources,
including image decryption keys, are denied even if CPU/GPU attestation succeeds.
Existing resource files are not authorization: keeping the image keys in KBS
storage does not keep their release rules. This also **does not revoke keys
already released to running guests** or erase a running guest's plaintext.
Stages 09–11 configure/check platform appraisal and health; they do not restore
workload release rules. Updating the scripts alone does not reset policy.

For a reference-only update, use 02 → 10 → 11 below, not stage 05. For an exposed
administrator credential, follow
[TRUSTEE-ADMIN-CREDENTIAL-SECURITY.md](TRUSTEE-ADMIN-CREDENTIAL-SECURITY.md),
not a blanket deployment rerun.

If intentionally rerunning stage 05, first pause new workload launches and
coordinate an exclusive maintenance window: no concurrent deployment, reference
update, or stage-12 installer. Stage 05 itself does not acquire the shared policy
update lock. Keep the following backup on secure services only; it contains
decryption keys and must never be handed to CoCo IT:

```bash
cd "$HOME/coco-service-admin"
BACKUP_DIR="$(mktemp -d "$HOME/trustee-policy-backup.XXXXXXXX")"
(
  source ./lib/common.sh
  lock_platform_reference_update
  install -m 0600 "$KBS_POLICY_DIR/resource-policy.rego" \
    "$BACKUP_DIR/resource-policy.rego"
  sudo cp -a "$KBS_STORAGE_DIR" "$BACKUP_DIR/kbs-storage"
  opa check --strict "$BACKUP_DIR/resource-policy.rego"
)
printf 'Private policy/resource backup: %s\n' "$BACKUP_DIR"
```

Do not continue if the backup fails. Record its exact directory, review the
policy and the releases it authorizes, and retain the authenticated handoffs and
their independently obtained manifest digests. A backup is not proof that the
policy was trustworthy. Keep maintenance exclusive until authorization is
restored; the backup lock alone does not cover subsequent commands.

When running stage 05 during this maintenance, wrap it in the same lock so it
cannot race the cooperating reference/workload installers:

```bash
(
  source ./lib/common.sh
  lock_platform_reference_update
  bash ./05-deploy-trustee.sh
)
```

After completing the required service maintenance and platform/health checks,
restore authorization by **one** of these methods:

1. **Reinstall each approved workload handoff (preferred).** Receive or retain the
   exact six-file handoff, including its 32-byte `image_key`, through a confidential
   authenticated channel. Reconfirm its manifest digest with the workload owner.
   If stage 12 previously removed the staging key, request the complete handoff
   again; do not synthesize a new key for an existing encrypted image. Follow
   [TRUSTED-HANDOFF-RUNBOOK.md](TRUSTED-HANDOFF-RUNBOOK.md), then repeat for each
   release. Stage 12 reviews and merges the policy, uploads/verifies all three
   resources, and commits the policy last. It does not reconstruct all releases
   automatically:

   ```bash
   EXPECTED_MANIFEST_SHA256='<independently authenticated SHA-256 of SHA256SUMS>'
   bash ./12-install-trusted-service-handoff.sh \
     "$HOME/incoming/RELEASE" "$EXPECTED_MANIFEST_SHA256"
   ```

2. **Restore the complete, still-approved policy from the private backup.** Use
   this only when the service administrator has reviewed the whole policy, all
   its releases remain approved, and the corresponding persisted resources are
   unchanged. The following comparison deliberately refuses restoration if the
   resource repository differs. Resolve any mismatch through authenticated
   handoff reinstallation instead of bypassing the check. Set `BACKUP_DIR` to
   the exact directory recorded above, then run:

   ```bash
   (
     source ./lib/common.sh
     lock_platform_reference_update
     opa check --strict "$BACKUP_DIR/resource-policy.rego"
     sudo diff --brief --recursive "$BACKUP_DIR/kbs-storage" "$KBS_STORAGE_DIR"
     kbs_admin set-resource-policy \
       --policy-file "$BACKUP_DIR/resource-policy.rego" >/dev/null
     cmp --silent "$BACKUP_DIR/resource-policy.rego" \
       "$KBS_POLICY_DIR/resource-policy.rego"
   )
   ```

   `set-resource-policy` replaces the entire global policy; it does not merge.
   Do not use this after adding new approvals during maintenance, since it would
   discard them. OPA syntax validation and byte comparison are not a security
   review and do not establish that the saved rules should still be trusted.

Finally, check service health, have CoCo IT launch a fresh authorized Pod, and
verify the new CPU/GPU appraisal and all three resource releases:

```bash
bash ./11-verify-service.sh
# After CoCo IT launches a fresh Pod for the restored release:
bash ./13-verify-workload-release.sh RELEASE 5m
```

Choose a log window containing that fresh launch, not an earlier successful run;
repeat the release check for each restored workload. Existing running Pods and a
passing health check alone do not prove that new key requests are authorized.
If maintenance changed public certificates, AS signing keys, platform references,
or workload inputs, update the dependent trust/configuration and regenerate any
affected handoffs before trying to reuse old Pod YAML.

### Later reference-only updates

The measurement field accepts one string or a list of 1–64 approved measurements.
Any exact measurement match is sufficient for the measurement check; the four
shared TCB floors and all other authorization checks still apply. Every update
replaces the complete list. See [MEASUREMENT-ALLOWLIST.md](MEASUREMENT-ALLOWLIST.md)
for the format, approval, transfer and verification procedure.

With services already configured, validate and approve the newly received file,
then run **02 → 10 → 11**, omitting `--configure-only`:

```bash
bash ./02-install-platform-reference-values.sh "$VALUES" --approve-platform-reference-values
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

Follow the five-value handoff guide for persistence verification. These updates
do not reinstall AS policies. Workload authorization uses
`12-install-trusted-service-handoff.sh`, followed by
`13-verify-workload-release.sh` after the authorized workload is launched.

Stage 12 still requires an interactive terminal. Its release-approval prompt
expires after 120 seconds; timeout or closed input cancels installation.
After installation succeeds, it prints the success message and receipt location
before optional staging-key removal. That prompt also expires after 120 seconds;
timeout or closed input retains the key and does not undo the installation.
No unattended approval mode is provided.

Follow the full guide for exact inputs and commands. The distributed TCB fields
are intentionally blank; stage 03 must not precede tool installation and
authenticated platform-input configuration on a fresh VM.

## Security and workload handoffs

- [PLATFORM-REFERENCE-VALUES-HANDOFF.md](PLATFORM-REFERENCE-VALUES-HANDOFF.md):
  the one-file, five-value trusted_system handoff; exact install/readback commands.
- [TRUSTED-HANDOFF-RUNBOOK.md](TRUSTED-HANDOFF-RUNBOOK.md): receive, authenticate
  and install the workload owner's six-file confidential service handoff.
- [SNP-TCB-POLICY.md](SNP-TCB-POLICY.md): measurement and minimum-TCB policy.
- [SECURITY.md](SECURITY.md): trust boundaries and threat model.
- [../trusted_system/README.md](../trusted_system/README.md): derive and export
  the five values for secure services and the separate admin launch contract. CoCo receives
  a public chart and digest pins from provisioning_node; no signed runtime bundle is used.

Never give CoCo the image key, publisher password, KBS admin token, Trustee TLS
private key, registry CA private key or registry server private key. CoCo
receives only public trust material and its public runtime/workload handoffs.

The registry starts empty; KBS denies workload-key release until an approved
workload-specific authorization is installed. Health checks alone are not
end-to-end attestation proof.

The `public/` directory contains instructions only. Generate this deployment's
certificates and distribute them only as documented.

For maintenance, see [the package boundary](../trusted_system/TEARDOWN.md).
