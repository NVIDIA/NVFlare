#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 2 && $# -le 3 && "$2" == --approve-platform-reference-values ]] || {
    echo "Usage: $0 VALUES.json --approve-platform-reference-values [--configure-only]" >&2; exit 2;
}
[[ $# == 2 || "$3" == --configure-only ]] || { echo 'Unknown option' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
PARSER="$SCRIPT_DIR/lib/platform-reference-values.py"
python3 "$PARSER" validate "$VALUES"
source "$SCRIPT_DIR/lib/common.sh"
lock_platform_reference_update
if [[ ${3:-} != --configure-only ]]; then
    require_sudo
    for file in "$KBS_CLIENT" "$ADMIN_TOKEN" "$TRUSTEE_PUBLIC_CERT" \
        "$KBS_POLICY_DIR/resource-policy.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego"; do need_file "$file"; done
    cmp --silent "$SCRIPT_DIR/policies/default_cpu.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" || \
        die 'The reviewed CPU policy must already be installed; follow the fresh-node guide first.'
    POLICY_BEFORE=$(sha256sum "$KBS_POLICY_DIR/resource-policy.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego")
fi
BACKUP=$(mktemp "$SCRIPT_DIR/platform.env.before-values.XXXXXX")
install -m 0600 "$SCRIPT_DIR/platform.env" "$BACKUP"
python3 "$PARSER" update-env "$VALUES" "$SCRIPT_DIR/platform.env"
printf 'Updated only the five reference fields in platform.env. Previous configuration: %s\n' "$BACKUP"
if [[ ${3:-} == --configure-only ]]; then
    printf 'Configuration-only: no RVPS, AS or KBS setting was changed.\n'
    exit 0
fi
source "$SCRIPT_DIR/lib/common.sh"
trap 'printf "Installation failed: references may be partially staged. Do not launch workloads; correct the failure and rerun this command. No automatic rollback is performed.\n" >&2' ERR
ids=(snp_min_reported_tcb_bootloader snp_min_reported_tcb_tee \
    snp_min_reported_tcb_snp snp_min_reported_tcb_microcode)
values=("$SNP_MIN_REPORTED_TCB_BOOTLOADER" "$SNP_MIN_REPORTED_TCB_TEE" \
    "$SNP_MIN_REPORTED_TCB_SNP" "$SNP_MIN_REPORTED_TCB_MICROCODE")
# Restrictive staging follows the existing stage-09 update order.
for key in "${ids[@]}"; do
    kbs_admin set-sample-reference-value "$key" 255 --as-integer --as-single-value >/dev/null
done
install_measurement_allowlist "$VALUES"
for i in "${!ids[@]}"; do
    kbs_admin set-sample-reference-value "${ids[i]}" "${values[i]}" \
        --as-integer --as-single-value >/dev/null
done
bash "$SCRIPT_DIR/10-verify-platform-reference-values.sh" "$VALUES"
[[ "$(sha256sum "$KBS_POLICY_DIR/resource-policy.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego")" == "$POLICY_BEFORE" ]] || \
    die 'An AS or KBS policy changed during reference installation; investigate before proceeding.'
printf 'Installed five RVPS references. AS CPU/GPU and KBS release-policy files are unchanged.\n'
