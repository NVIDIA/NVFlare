# Approved GPU driver and VBIOS references

GPU releases require an independently approved version allowlist in Trustee
RVPS. This is separate from the SNP and TDX CPU reference handoffs, so an
update for one attester does not replace the other's values. The pinned Trustee
v0.22.0 GPU policy queries `allowed_driver_versions` and
`allowed_vbios_versions`. The service kit installs a reviewed policy derived
from Trustee commit `512fed65642015b849f38fb13bfdec7806639987`
(`attestation-service/src/ear_token/ear_default_policy_gpu.rego`); it preserves all NVIDIA NRAS, certificate,
report, RIM, secure-boot, debug and version checks, and emits the three signed
trust claims required by the KBS and NVFlare.

## Collect and approve values

The trusted GPU/platform security owner should retain fresh, authenticated
NVIDIA attestation evidence from the intended GPU fleet. Review the signed NRAS
claims `x-nvidia-gpu-driver-version` and `x-nvidia-gpu-vbios-version` against
the approved driver, firmware, GPU model and change record. Record the exact
strings and evidence digests in the private approval record. Do not derive an
approval automatically from `nvidia-smi`, a pod log, or values supplied by the
workload owner; those diagnostics do not authenticate the NRAS appraisal.
Approval and authenticated transfer to secure services remain separate
administrative decisions. No production values or private evidence are in
the role kit.

Send one private JSON file with exactly these fields:

```json
{
  "schema": "coco-gpu-reference-values/v1",
  "allowed_driver_versions": ["reviewed-driver-version"],
  "allowed_vbios_versions": ["reviewed-vbios-version"]
}
```

Each list must contain 1–64 unique, exact version strings. Replacing a list
removes omitted versions from future appraisals. The validator enforces shape
and syntax, not authenticity or suitability of a version. Avoid copying
diagnostic sample strings into an actual deployment approval.

## Configure, install, and verify

First run normal service stages 01–09, including CPU policy/reference setup.
On secure services, authenticate the approved GPU JSON transfer and review it:

```bash
cd /home/service_operator/coco-service-admin
GPU_VALUES=/home/service_operator/incoming-platform/gpu-reference-values.json
python3 ./lib/gpu-reference-values.py validate "$GPU_VALUES"
bash ./02g-install-gpu-reference-values.sh "$GPU_VALUES" \
  --approve-gpu-reference-values --configure-only
bash ./02g-install-gpu-reference-values.sh "$GPU_VALUES" \
  --approve-gpu-reference-values
bash ./10g-verify-gpu-reference-values.sh "$GPU_VALUES" --restart-rvps
bash ./11-verify-service.sh
```

Configuration saves a mode-0600 approved snapshot and its path in
`platform.env`; previous configuration is backed up. The installer holds
the shared administrator lock and checks that the active GPU policy is the
reviewed policy. It clears driver approval and reads back the empty list,
replaces and reads back VBIOS versions, then activates and reads back driver
versions. This prevents a completed partial update from approving a mixed
old/new pair. If a write or readback fails, it attempts to clear driver
approval again and reports whether that compensation was verified. If
compensation cannot be verified, treat GPU approval state as unknown: stop GPU
release work and inspect RVPS. A process killed before cleanup may also leave
an unknown state. The installer does not modify CPU references or KBS release
policy.

Stage 11 verifies the configured GPU lists and installed policy. Stage 12
checks those same live values under its shared admin lock before publishing
any GPU release. CPU-only handoffs do not require GPU lists. Existing issued
attestation tokens and keys already released to running guests are not
immediately revoked by a reference update; coordinate workload maintenance
and normal token expiry when changing an active GPU baseline.
