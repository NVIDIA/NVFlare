# TDX platform references: approve, install and verify

The trusted platform authority sends one authenticated
`platform-reference-values.json`, coordinated through the provisioning node.
Secure services receives no quote archive, Kata artifacts, private keys or AS
policy in this handoff. The service administrator decides whether to trust the
sender and proposed platform. Schema validation alone is not that approval.
Never approve values supplied only by the adversarial CoCo cluster operator.

Use [SERVICE-INSTALLATION.md](SERVICE-INSTALLATION.md) for service packages,
TLS, registry, audience hardening and isolation. The instructions below supply
the TDX-specific reference/policy steps; they do not authorize any image key.

## Handoff schema and approval meaning

The exact top-level fields are:

```json
{
  "schema": "coco-platform-reference-values/v2",
  "tee": "tdx",
  "profiles": []
}
```

This is a schema illustration, **not a valid approval file**: `profiles` must
contain 1–64 verified, approved profiles. Each profile has exactly:

| Field | Required value |
| --- | --- |
| `id` | Unique 1–64 character identifier, starting with a lowercase letter/digit, then lowercase letters, digits, `_`, `.` or `-` |
| `mr_td` | 96 lowercase hexadecimal characters |
| `rtmr_1` | 96 lowercase hexadecimal characters |
| `rtmr_2` | 96 lowercase hexadecimal characters |
| `xfam` | 16 lowercase hexadecimal characters |
| `tdvfkernel` | 96 lowercase hexadecimal characters, verified kernel event digest |
| `tdvfkernelparams` | 96 lowercase hexadecimal characters, verified kernel-parameter event digest |

Use the trusted-system stage-10 export; do not fill this schema with dummy
measurements. Duplicate IDs/tuples, malformed hex, unknown fields or missing
fields fail validation. The AS policy requires a match to **one whole tuple**,
not independent field allowlists. This prevents accidentally accepting a
combination assembled from different approved launches. The quote verifier
and AS configuration/TCB checks remain required in addition to matching references.

The entire profile array is registered under one RVPS reference ID,
`coco_tdx_profiles_v2`. Each TDX install replaces that complete array; include
every previously approved profile that must remain valid. Removing one revokes
its reference match for future appraisals, not keys or tokens already issued.
SNP's separate five references are not removed by a TDX install.

## 1. Validate the authenticated file without changing services

From the secure-services kit:

```bash
cd "$HOME/coco-service-admin"
VALUES="$HOME/incoming-platform/platform-reference-values.json"
python3 ./lib/platform-reference-values.py validate "$VALUES"
```

Review the entire tuple set and its provenance with the trusted platform
authority. Do not accept a replacement hash or platform file from CoCo IT.

## 2. Fresh node: install the reviewed policy before reference updates

Prepare `platform.env` from its template and review the host, TLS endpoints,
registry publisher network and private state paths as in the installation
guide. Then run the existing numeric stages in order:

```bash
bash ./01-install-host-tools.sh
bash ./02-install-platform-reference-values.sh "$VALUES" \
  --approve-platform-reference-values --configure-only
bash ./03-preflight.sh
bash ./04-build-trustee-main.sh
bash ./05-deploy-trustee.sh
bash ./06-configure-trustee-tls.sh
bash ./07-harden-kbs-admin-audience.sh
bash ./08-deploy-private-registry.sh
bash ./09-install-platform-policy.sh --approve-pinned-platform
bash ./10-verify-platform-reference-values.sh "$VALUES" --restart-rvps
bash ./11-verify-service.sh
```

Stage 02's `--configure-only` does not contact RVPS. It saves the selected
approved JSON in `approved-platform-reference-values.json` with mode 0600 and
records `PLATFORM_REFERENCE_VALUES_FILE` in `platform.env`. Stage 09 installs
the service kit's reviewed combined SNP/TDX AS CPU policy and then the selected
references. It does not install or replace KBS workload authorization or the
GPU policy. Use `--approve-pinned-platform` for TDX; the legacy
`--approve-pinned-snp-platform` flag authorizes SNP inputs only.

The AS policy comes from this trusted service kit, not from the platform
handoff. Read its SNP and TDX rules before the explicit approval command.
The TDX path requires its approved kernel/event-log route and exact successful
CPU trust vector `(3,2,2)`; do not enable a permissive fallback to get a token.

## 3. Existing services: one-time policy migration

When moving from the old SNP-only CPU policy to this reviewed combined policy,
preserve current configuration, reference files and policy backups privately.
Review the new policy and coordinate the change. **Do not rerun stage 05**;
it overwrites workload authorization with default-deny.

With the new kit and an authenticated TDX reference file, run:

```bash
bash ./02-install-platform-reference-values.sh "$VALUES" \
  --approve-platform-reference-values --configure-only
bash ./09-install-platform-policy.sh --approve-pinned-platform
bash ./10-verify-platform-reference-values.sh "$VALUES" --restart-rvps
bash ./11-verify-service.sh
```

The configured file denotes the most recently selected TEE, not the complete
contents of RVPS. Keep and independently verify the prior SNP JSON too:

```bash
SNP_VALUES="$HOME/approved-platforms/snp-platform-reference-values.json"
bash ./10-verify-platform-reference-values.sh "$SNP_VALUES"
```

Do not infer preservation merely from a TDX success message. Test an approved
existing SNP workload after the policy change, as well as the TDX workload.

## 4. Later reference-only changes

The new combined CPU policy must already be active. After approving the complete
replacement TDX profile set, use **02 → 10 → 11**:

```bash
bash ./02-install-platform-reference-values.sh "$VALUES" \
  --approve-platform-reference-values
bash ./10-verify-platform-reference-values.sh "$VALUES" --restart-rvps
bash ./11-verify-service.sh
```

The updater checks the active reviewed CPU policy before changing references,
uses authenticated TLS administration, replaces the complete TDX tuple set in
one RVPS write, reads it back, and confirms appraisal/release policies did not
change. It does not accept an old SNP-only CPU policy as TDX support. Do not
manually register independent `mr_td`, `rtmr_1`, or kernel allowlists instead.

Stage 10 queries live RVPS and requires exact full-set equality, including
types and tuple membership. `--restart-rvps` deliberately restarts RVPS and
performs another live readback to verify persistence. Perform it in a suitable
maintenance window. A passing check does not prove a fresh quote or key release.
On failure stop and investigate; do not lower policy or approve unexpected values.

To host both SNP and TDX, run the applicable installer/verifier for each
independently approved JSON. Each TEE update replaces only its own reference
set. Neither update removes the other TEE's references or authorizes a workload.

## 5. Approve a workload separately

Receive the workload owner's confidential six-file `trusted-service/` handoff,
authenticate its manifest digest, and follow
[TRUSTED-HANDOFF-RUNBOOK.md](TRUSTED-HANDOFF-RUNBOOK.md). Its v2 authorization
names the CPU type, GPU requirement, three resource paths and expected InitData.
TDX-only must require CPU appraisal; TDX+GPU must also require NVIDIA appraisal.
Successful CPU reference installation alone releases no decryption key.

After CoCo launches a fresh authorized Pod, verify the corresponding appraisals
and resource requests, then confirm authenticated workload execution and
CoCoAuthorizer proof verification by an ordinary server. Preserve actual live
positive/negative test results privately; this runbook is not itself evidence
that hardware validation has succeeded.
