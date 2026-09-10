# Approve several SNP launch measurements

These commands require this complete runnable service kit and its private
configuration, not just a copy of this document.

The existing AS CPU policy already checks membership:

```rego
executables := 3 if {
    input.snp
    input.snp.measurement in query_reference_value("snp_launch_measurement")
}
```

RVPS stores one array under `snp_launch_measurement`. Matching any entry passes
this measurement check. The four shared minimum reported-TCB checks, debug
and migration restrictions, GPU appraisal, and KBS workload/resource-path
authorization still have to pass. No AS or KBS policy change is needed for a
measurement-only update through stage 02.

## 1. Prepare the complete approved file

The trusted coordinator on **provisioning_node** collects authenticated exports from
the trusted systems. The service administrator reviews every measurement and
one common set of four floors. Preserve the original exports for audit.
Do not derive trusted references from adversarial CoCo IT.

Keep exactly these five fields. The following is a **schema illustration, not
valid install input**: replace the placeholders with independently approved
96-character lowercase hexadecimal measurements and integer floors.

```text
{
  "snp_launch_measurement": ["<approved measurement A>", "<approved measurement B>"],
  "snp_min_reported_tcb_bootloader": <approved integer 0..255>,
  "snp_min_reported_tcb_tee": <approved integer 0..255>,
  "snp_min_reported_tcb_snp": <approved integer 0..255>,
  "snp_min_reported_tcb_microcode": <approved integer 0..255>
}
```

Use 1–64 unique measurements. Existing files with a single measurement string
remain supported; the trusted-system exporter need not change. For an update,
include **all** measurements that should remain approved. The installer replaces
the list; submitting only a new measurement removes the old ones. It never
automatically merges lists or lowers TCB floors.

These are shared floors, not per-measurement tuples. If different measurements
need different TCB requirements, this format does not express that association;
use a separately reviewed profile-aware policy design. Do not flatten those
profiles into independent allowlists or select the lowest floor from each.

## 2. Transfer through provisioning_node

After reviewing the complete file and authenticating the SSH host key, run on
**provisioning_node**, adjusting `VALUES` to the reviewed local file:

```bash
VALUES=/absolute/path/to/reviewed/platform-reference-values.json
SERVICE_SSH=service_operator@secure-services.example.com
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$SERVICE_SSH" \
  'install -d -m 0700 /home/service_operator/incoming-platform'
scp -o BatchMode=yes -o StrictHostKeyChecking=yes "$VALUES" \
  "$SERVICE_SSH:/home/service_operator/incoming-platform/platform-reference-values.next.json"
sha256sum "$VALUES"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$SERVICE_SSH" \
  'sha256sum /home/service_operator/incoming-platform/platform-reference-values.next.json'
```

Both hashes must match. Use a new destination filename if `.next.json` already
holds a handoff you need to preserve. Transfer no private keys or KBS admin token.

## 3. Review and install on the service machine

For **already configured** secure services, run as `service_operator` in this order:

```bash
cd /home/service_operator/coco-service-admin
VALUES=/home/service_operator/incoming-platform/platform-reference-values.next.json
python3 ./lib/platform-reference-values.py validate "$VALUES"
# Stop here until the service owner has approved the entire list and floors.
bash ./02-install-platform-reference-values.sh "$VALUES" --approve-platform-reference-values
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

Stage 02 retains a private `platform.env` backup, stages restrictive floors,
replaces the list in one TLS-authenticated admin request, restores the approved
floors, then reads all five references back. It checks that AS CPU/GPU and KBS
policy files remain unchanged. The verifier requires the **entire** measurement
set to match, not merely the presence of one approved entry. List ordering does
not matter; extra, missing, duplicate or incorrectly typed entries fail.

The environment variable retains its name. A list is stored safely as:

```text
SNP_LAUNCH_MEASUREMENT='["<measurement A>","<measurement B>"]'
```

Use the JSON installer to populate it; do not paste the illustrative placeholders.
Stages 03, 09 and 11 understand both the old scalar and the new array form.

For a **fresh machine**, follow [SERVICE-INSTALLATION.md](SERVICE-INSTALLATION.md):
stage 02 with `--configure-only`, then stages 03–09, 10 and 11. Stage 09 consumes
the full list from `platform.env`, so it does not collapse it to one measurement.

## 4. Verify persistence and authorization

During an approved maintenance window, restart RVPS and reread the values:

```bash
source ./lib/common.sh
sudo docker compose -p "$TRUSTEE_PROJECT" -f "$TRUSTEE_COMPOSE" restart rvps
# Wait for RVPS readiness; if needed, rerun the read-only verifier when ready.
bash ./10-verify-platform-reference-values.sh "$VALUES"
bash ./11-verify-service.sh
```

Require all five reference checks to pass again. For end-to-end acceptance,
rehearse an authorized workload under each approved launch profile; an
unlisted measurement must fail the measurement appraisal. A listed measurement
with inadequate TCB, failed GPU appraisal or mismatched workload identity must
still be denied. Readback and unit tests are not proof of live key-release behavior.

Five-key updates are not transactional. Coordinate writers and pause new
workload launches. The installers' lock covers only this service-kit directory;
it cannot serialize other API clients. TCB staging at 255 is restrictive, not
an unconditional deny. On failure, investigate and rerun the same approved
input; there is no automatic rollback. Do not lower floors to bypass an error.

Local regression tests (isolated mock backend, no live writes):

```bash
PYTHONDONTWRITEBYTECODE=1 coco_deployment/.venv/bin/python -m unittest discover \
  -s coco_deployment/service/tests -v
```
