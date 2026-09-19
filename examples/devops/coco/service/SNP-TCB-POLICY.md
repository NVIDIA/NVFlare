# SNP minimum reported-TCB policy

## Security objective

A valid SNP report signature proves that AMD signed the reported claims. It does
not by itself say that the signed firmware or microcode level meets this
service's security baseline. The service therefore requires both:

- an exact match to one approved launch measurement extracted from a trusted system's
  challenge-bound, AMD-signed rehearsal report and delivered by the trusted
  sender over an authenticated channel; and
- independently approved minimum values for the four numeric Trustee SNP
  `reported_tcb_*` claims.

The installer supports one measurement string or a list of 1–64 unique approved
measurements. All share the same four TCB floors; it does not express
measurement-specific baselines. See [MEASUREMENT-ALLOWLIST.md](MEASUREMENT-ALLOWLIST.md).

Trustee documents the reported TCB as the policy-facing value. The actual
firmware can be newer than the reported value while a provider rolls out
provisional firmware, so this policy intentionally checks reported TCB rather
than attempting to infer current or committed TCB fields that Trustee does not
export to Rego.

## Required approval inputs

The platform security owner—not the workload owner and not CoCo IT—sets these in
`platform.env`:

```bash
SNP_MIN_REPORTED_TCB_BOOTLOADER="DECIMAL_UINT8"
SNP_MIN_REPORTED_TCB_TEE="DECIMAL_UINT8"
SNP_MIN_REPORTED_TCB_SNP="DECIMAL_UINT8"
SNP_MIN_REPORTED_TCB_MICROCODE="DECIMAL_UINT8"
```

Each value must be decimal `0..255`. There is deliberately no example numeric
baseline: copying one would turn a platform-specific security decision into an
unsafe default. Establish the floors from the approved server model, AMD
security guidance and the organization's firmware/microcode patch baseline.
Record the source, approver, date, affected CPU generation, and any accepted
advisories outside this repository.

An attestation report collected from `coco` can confirm whether that cluster
meets an already approved floor. It must not define the floor.

## Installation

The current trusted_system handoff is exactly one JSON file with five values. Follow
[PLATFORM-REFERENCE-VALUES-HANDOFF.md](PLATFORM-REFERENCE-VALUES-HANDOFF.md)
for export, transfer, approval, reference-only installation with stage 02,
and live read-back verification with stage 10. No signed archive or AS policy
is transferred from trusted_system. The commands below are for the initial installation
of secure services' own reviewed CPU policy, not routine reference-only updates.

After editing and independently reviewing `platform.env`, display the script and
then run:

```bash
./03-preflight.sh
./09-install-platform-policy.sh --approve-pinned-snp-platform
./11-verify-service.sh
```

Stage 09 writes each floor into RVPS as a numeric single value. The Rego policy
uses `is_number` before every `>=` comparison, so a missing reference (`null`),
a quoted string, or an array cannot accidentally authorize a report. The stage
temporarily writes `255` to all four references before enabling the policy,
making an interrupted update restrictive rather than permissive.

The KBS resource policy separately requires exactly the CPU and GPU submodules,
and compares each complete signed EAR trust vector with the platform-approved
vector. Therefore a below-floor CPU appraisal changes the CPU trust vector and
cannot release the image key, signature policy, or public verification key for
any workload release. Requiring exact submodule cardinality also rejects an
unexpected third appraisal rather than silently ignoring it.

## Rotation

Raise a floor when the security baseline drops support for an older TCB. Never
lower a floor merely to restore a failing adversarial cluster. A lower value is
a reviewed security exception and requires the same independent approval as a
new platform.

After any change:

1. install the approved five-value file with stage 02, then verify with stages
   10 and 11 (use stage 09 only when deliberately installing the reviewed CPU policy);
2. launch an authorized workload and verify successful composite attestation;
3. verify that a synthetic or captured below-floor claim changes the CPU trust
   vector away from the approved value and produces no KBS resource response;
4. record the approved inputs and results in an installation-specific audit
   record outside the distributed kit, only after retaining live evidence.

## Upstream claim contract

The pinned Trustee source documents these claims in
`attestation-service/docs/tcb_claims.md` and emits them from
`deps/verifier/src/snp/mod.rs`. Revalidate the names, numeric types, and RVPS
extension behavior before changing the pinned Trustee commit.
