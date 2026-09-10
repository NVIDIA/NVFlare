# Service administrator kit

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
