#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 && "$1" == "--approve-pinned-snp-platform" ]] || {
    printf 'Usage: %s --approve-pinned-snp-platform\n' "$0" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
require_sudo
need_file "${KBS_CLIENT}"
need_file "${TRUSTEE_PUBLIC_CERT}"
need_file "${ADMIN_TOKEN}"
need_file "${SCRIPT_DIR}/policies/default_cpu.rego"
need_file "${KBS_POLICY_DIR}/resource-policy.rego"

lock_platform_reference_update
PLATFORM_VALUES_JSON="$(configured_platform_values)"
VALUES_FILE="$(mktemp)"
trap 'rm -f -- "$VALUES_FILE"' EXIT
printf '%s\n' "$PLATFORM_VALUES_JSON" > "$VALUES_FILE"
require_uint8 SNP_MIN_REPORTED_TCB_BOOTLOADER "${SNP_MIN_REPORTED_TCB_BOOTLOADER-}"
require_uint8 SNP_MIN_REPORTED_TCB_TEE "${SNP_MIN_REPORTED_TCB_TEE-}"
require_uint8 SNP_MIN_REPORTED_TCB_SNP "${SNP_MIN_REPORTED_TCB_SNP-}"
require_uint8 SNP_MIN_REPORTED_TCB_MICROCODE "${SNP_MIN_REPORTED_TCB_MICROCODE-}"
[[ "$(stat -c '%a' "${ADMIN_TOKEN}")" == 600 ]] \
    || die "KBS admin token must be mode 0600"

printf 'Installing independently approved SNP platform policy:\n'
printf '  launch measurement:       %s\n' "${SNP_LAUNCH_MEASUREMENT}"
printf '  minimum bootloader SVN:   %s\n' "${SNP_MIN_REPORTED_TCB_BOOTLOADER}"
printf '  minimum TEE SVN:          %s\n' "${SNP_MIN_REPORTED_TCB_TEE}"
printf '  minimum SNP firmware SVN: %s\n' "${SNP_MIN_REPORTED_TCB_SNP}"
printf '  minimum microcode SVN:    %s\n' "${SNP_MIN_REPORTED_TCB_MICROCODE}"

# Stage restrictive floors before enabling the reviewed policy. This is not an
# unconditional deny or an atomic transaction: coordinate updates and pause
# new launches. An older policy may ignore these floors until replaced.
for reference_id in \
    snp_min_reported_tcb_bootloader \
    snp_min_reported_tcb_tee \
    snp_min_reported_tcb_snp \
    snp_min_reported_tcb_microcode; do
    kbs_admin set-sample-reference-value "${reference_id}" 255 \
        --as-integer --as-single-value >/dev/null
done

kbs_admin set-attestation-policy --type rego --id default_cpu \
    --policy-file "${SCRIPT_DIR}/policies/default_cpu.rego" >/dev/null

install_measurement_allowlist "$VALUES_FILE"

TCB_REFERENCE_IDS=(
    snp_min_reported_tcb_bootloader
    snp_min_reported_tcb_tee
    snp_min_reported_tcb_snp
    snp_min_reported_tcb_microcode
)
TCB_REFERENCE_VALUES=(
    "${SNP_MIN_REPORTED_TCB_BOOTLOADER}"
    "${SNP_MIN_REPORTED_TCB_TEE}"
    "${SNP_MIN_REPORTED_TCB_SNP}"
    "${SNP_MIN_REPORTED_TCB_MICROCODE}"
)
for index in "${!TCB_REFERENCE_IDS[@]}"; do
    kbs_admin set-sample-reference-value \
        "${TCB_REFERENCE_IDS[index]}" "${TCB_REFERENCE_VALUES[index]}" \
        --as-integer --as-single-value >/dev/null
done

need_file "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego"
need_file "${AS_STORAGE_DIR}/attestation_service_policy/default_gpu.rego"
cmp --silent "${SCRIPT_DIR}/policies/default_cpu.rego" \
    "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego" \
    || die "persisted CPU policy differs from the reviewed policy"

bash "$SCRIPT_DIR/10-verify-platform-reference-values.sh" "$VALUES_FILE"

curl --fail --silent --show-error --cacert "${TRUSTEE_PUBLIC_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"
printf 'Platform CPU policy, SNP measurement, and minimum reported TCB floors installed.\n'
printf 'The pinned post-v0.21 Trustee default GPU appraisal policy remains active.\n'
printf 'The KBS resource policy remains default-deny until release fragments are merged.\n'
