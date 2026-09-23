# trusted_system → secure services: five platform-reference values

First configure the complete service kit as described in
[../CONFIGURATION.md](../CONFIGURATION.md). No approved values are bundled.

The secure services owner trusts the platform inputs produced by `trusted_system`. The handoff
is **one file, `platform-reference-values.json`, containing exactly five
fields**. The measurement field accepts either one measurement or an approved
allowlist; the other four fields remain shared minimum TCB integers.
Do not send Kata artifacts, SNP reports, signatures, manifests,
signing keys, or AS policy files in this handoff. Keep the evidence on
`trusted_system`. CoCo's separate runtime setup uses the public chart and digest pins
from provisioning_node; it does not require a signed bundle.

This file is not signed. Its authenticity depends on the trusted sender and
authenticated transfer. Do not accept replacement values from the adversarial
CoCo owner. Shape/type validation is not proof of trustworthy measurements or
firmware. No workload keys or workload-release authorization are included.

## File contents and destination

| JSON key / RVPS reference ID | Type | Corresponding field in secure services' `platform.env` |
|---|---|---|
| `snp_launch_measurement` | One 96-character lowercase hex string, or a list of 1–64 unique such strings | `SNP_LAUNCH_MEASUREMENT` |
| `snp_min_reported_tcb_bootloader` | Integer 0..255 | `SNP_MIN_REPORTED_TCB_BOOTLOADER` |
| `snp_min_reported_tcb_tee` | Integer 0..255 | `SNP_MIN_REPORTED_TCB_TEE` |
| `snp_min_reported_tcb_snp` | Integer 0..255 | `SNP_MIN_REPORTED_TCB_SNP` |
| `snp_min_reported_tcb_microcode` | Integer 0..255 | `SNP_MIN_REPORTED_TCB_MICROCODE` |

Zero is a valid floor. Strings containing numbers, booleans, duplicate keys,
extra fields and missing fields are rejected. Empty lists, duplicate
measurements and malformed measurements are rejected. Input files are limited
to 8192 bytes. Existing single-string handoffs remain valid.

Every install **replaces the complete measurement allowlist**; it does not
append or merge with RVPS. Include every previously approved measurement that
should remain accepted. Omitting one revokes its measurement match for future
appraisals (it cannot recover keys already released). Review the full list
before approval. See [MEASUREMENT-ALLOWLIST.md](MEASUREMENT-ALLOWLIST.md) for
preparing and installing a multiple-measurement file.

AS requires an exact match to **any one** measurement in the list and uses the four
floors for numeric `reported_tcb_* >= minimum` checks. These are CPU platform
references, not GPU reference values. Existing GPU verification and the KBS
requirement for acceptable CPU **and** GPU results remain in place; installing
these values alone does not authorize any workload's resource paths or keys.
Do not lower a floor merely to make a failing cluster pass; a reduction requires
an explicitly reviewed security exception.
All measurements use the **same four floors**. This is not a set of
measurement-specific TCB profiles; do not combine different profiles by
automatically choosing their lowest floors.

## 1. trusted_system: export after stage 09

Complete trusted-system stages 01 through 09 in numeric order (installation
stages are for a fresh machine). Stage 09 verifies the AMD evidence and writes
`platform-reference.final.env`. The exporter uses its approved report-derived
measurement, not the diagnostic offline measurement.

Run on **trusted_system**, replacing the private configuration path if needed:

```bash
cd /home/operator/coco_deployment
source /home/operator/private-platform-reference/platform-reference.env
FINAL_ENV="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE/platform-reference.final.env"
install -d -m 0700 "$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE/service-out"
VALUES="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE/service-out/platform-reference-values.json"
bash trusted_system/10-export-platform-reference-values.sh "$FINAL_ENV" "$VALUES"
```

The output file must not already exist; preserve prior exports and use a new
output directory if necessary. The exporter does not independently repeat
attestation verification; its input must be the trusted stage-09 result.
It copies only the five values. No signed service bundle is needed.

## 2. Transfer exactly that file

After independently authenticating secure services' SSH host key, run on **trusted_system**:

```bash
SECURE_SERVICES_SSH=service_operator@secure-services.example.com
ssh -o StrictHostKeyChecking=yes "$SECURE_SERVICES_SSH" \
  'install -d -m 0700 /home/service_operator/incoming-platform'
scp -o StrictHostKeyChecking=yes "$VALUES" \
  "$SECURE_SERVICES_SSH:/home/service_operator/incoming-platform/platform-reference-values.json"
```

secure services' SSH allowlist and account authorization must permit the trusted sender.
If direct SSH is not available, a trusted coordinator can transfer the same
single file over authenticated connections. Do not route it through CoCo or
add files to this handoff to compensate for missing SSH access.

## 3. secure services: validate and review without changing settings

Run as **service_operator**, with the current service kit at `~/coco-service-admin`:

```bash
cd /home/service_operator/coco-service-admin
VALUES=/home/service_operator/incoming-platform/platform-reference-values.json
python3 ./lib/platform-reference-values.py validate "$VALUES"
```

This prints the five validated values. It changes no files or services.
Confirm the file came from the trusted `trusted_system` operator and approve its
applicability to the intended CPU/Kata platform. Never fill missing fields with
guessed values to get past validation.

## 4. secure services: install into an already-configured Trustee

Prerequisites: KBS, AS, RVPS and Trustee TLS are running; the KBS client,
local admin token and public TLS certificate are present; the reviewed
`policies/default_cpu.rego` is already the active AS CPU policy.

```bash
cd /home/service_operator/coco-service-admin
bash ./02-install-platform-reference-values.sh "$VALUES" \
  --approve-platform-reference-values
```

Execution order inside stage 02:

1. Validate the JSON before any changes and require the reviewed CPU policy.
2. Save a mode-0600 copy of the previous `platform.env`; replace only its five
   reference fields, preserving all other service configuration.
3. Temporarily register the four TCB floors as 255, following stage 09's
   restrictive update order.
4. Replace `snp_launch_measurement` with the complete approved JSON array in
   one authenticated HTTPS request (a single string becomes a one-item array).
5. Register the four approved TCB floors as scalar integers.
6. Read all five references back through the authenticated KBS admin API and
   compare them with the received JSON, requiring exact measurement-set
   equality regardless of order, then confirm AS CPU/GPU and KBS
   release-policy file hashes are unchanged.

The underlying administration commands are:

```bash
source ./lib/common.sh
# The installer sets all four temporary numeric floors before this command.
install_measurement_allowlist "$VALUES"
kbs_admin set-sample-reference-value snp_min_reported_tcb_bootloader "$SNP_MIN_REPORTED_TCB_BOOTLOADER" --as-integer --as-single-value
kbs_admin set-sample-reference-value snp_min_reported_tcb_tee "$SNP_MIN_REPORTED_TCB_TEE" --as-integer --as-single-value
kbs_admin set-sample-reference-value snp_min_reported_tcb_snp "$SNP_MIN_REPORTED_TCB_SNP" --as-integer --as-single-value
kbs_admin set-sample-reference-value snp_min_reported_tcb_microcode "$SNP_MIN_REPORTED_TCB_MICROCODE" --as-integer --as-single-value
```

Use stage 02 for the guarded sequence rather than executing selected writes
manually. `install_measurement_allowlist` uses Python's standard-library HTTPS
client with the configured CA certificate, hostname validation and the local
KBS admin token file. It posts a native array using Trustee's sample extractor
envelope to `/kbs/v0/reference-value`; it does not follow redirects or expose
the token in command arguments. The sample extractor is not provenance
verification: sender approval and the authenticated admin channel are essential.
Do **not** pass a serialized array to `kbs-client set-sample-reference-value`:
that CLI would register the array text as one string, not multiple measurements.

Updates across all five references are not an atomic RVPS transaction and may
temporarily deny requests. Stages 02 and 09 share a nonblocking local lock;
another copy of the kit or a manual API caller is not covered by that lock.
Staging TCB floors at 255 is restrictive, not an unconditional deny rule.
Do not install references concurrently or launch new workloads
during installation. On failure, stop, inspect the error, and rerun the same
approved file after fixing the cause. Do not assume automatic rollback or
restore permissive reference values as a workaround.

Stage 02 does **not** install an AS policy, change KBS workload-release rules,
copy an image key, modify TLS, or restart a container. A CPU-policy mismatch
causes it to stop before configuration/reference changes.

## 5. secure services: verify the values and their persistence

```bash
bash ./10-verify-platform-reference-values.sh "$VALUES"
```

Require five `PASS` lines naming the reference ID and exact expected value,
followed by `All five live RVPS references match the received file.` This
queries the live service, not merely `platform.env` or files on disk. Missing,
mismatched or wrongly typed reference results cause a nonzero exit status.
For the measurement array, missing, extra or duplicate entries also fail;
reordering the same unique entries is allowed.
The verifier is read-only.

Then test persistence on **secure services** (this intentionally restarts only RVPS):

```bash
source ./lib/common.sh
sudo docker compose -p "$TRUSTEE_PROJECT" -f "$TRUSTEE_COMPOSE" restart rvps
sleep 3
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

All checks must pass before treating the platform as configured. If the restart
needs longer, wait for RVPS readiness and rerun verification; do not rewrite
the values merely because a readiness probe failed.

## Fresh secure services node: where this fits

Use [SERVICE-INSTALLATION.md](SERVICE-INSTALLATION.md) for host tools and
service creation. After receiving and approving the five-value file, run
stage 02 with `--configure-only` to prepare `platform.env` for preflight; this
does not call KBS or RVPS. After the services are running, run the existing
stage 09 once to install secure services' reviewed CPU policy and the five references,
then stage 10 and stage 11. The AS policy comes from secure services' reviewed service
kit, **not** from the trusted_system handoff.

For later reference-only updates, use steps 3–5 above. The legacy signed
platform-bundle verifier and installer have been removed from the service kit.
Send only the five-value JSON file, not a signed service bundle.
