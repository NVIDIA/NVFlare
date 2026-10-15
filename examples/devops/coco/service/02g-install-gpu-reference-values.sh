#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

[[ $# -ge 2 && $# -le 3 && "$2" == --approve-gpu-reference-values ]] || {
    echo "Usage: $0 VALUES.json --approve-gpu-reference-values [--configure-only]" >&2; exit 2;
}
[[ $# == 2 || "$3" == --configure-only ]] || { echo 'Unknown option' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
PARSER="$SCRIPT_DIR/lib/gpu-reference-values.py"
python3 "$PARSER" validate "$VALUES"
source "$SCRIPT_DIR/lib/common.sh"
lock_platform_reference_update

if [[ ${3:-} != --configure-only ]]; then
    require_sudo
    for file in "$KBS_CLIENT" "$ADMIN_TOKEN" "$TRUSTEE_PUBLIC_CERT" \
        "$KBS_POLICY_DIR/resource-policy.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego"; do need_file "$file"; done
    cmp --silent "$SCRIPT_DIR/policies/default_gpu.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego" || \
        die 'Reviewed GPU policy must be installed before GPU references'
    POLICY_BEFORE=$(sha256sum "$KBS_POLICY_DIR/resource-policy.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
        "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego")
fi

BACKUP=$(mktemp "$SCRIPT_DIR/platform.env.before-gpu-values.XXXXXX")
install -m 0600 "$SCRIPT_DIR/platform.env" "$BACKUP"
if [[ -n ${GPU_REFERENCE_VALUES_FILE:-} ]]; then
    [[ -f "$GPU_REFERENCE_VALUES_FILE" && -r "$GPU_REFERENCE_VALUES_FILE" ]] || \
        die 'Previous GPU reference snapshot must be a readable regular file'
    install -m 0600 "$GPU_REFERENCE_VALUES_FILE" "$BACKUP.references.json"
fi
python3 "$PARSER" update-env "$VALUES" "$SCRIPT_DIR/platform.env"
printf 'Saved approved GPU reference snapshot. Previous configuration: %s\n' "$BACKUP"
if [[ ${3:-} == --configure-only ]]; then
    printf 'Configuration-only: RVPS and AS policies were not changed.\n'
    exit 0
fi

source "$SCRIPT_DIR/lib/common.sh"
[[ -n ${GPU_REFERENCE_VALUES_FILE:-} ]] || die 'Approved GPU reference snapshot was not configured'
VALUES=$(realpath -e -- "$GPU_REFERENCE_VALUES_FILE")
python3 "$PARSER" validate "$VALUES"
verify_stage() {
    local stage=$1 reference_id raw
    reference_id=$(python3 "$PARSER" reference-id "$VALUES" "$stage")
    raw=$(kbs_admin get-reference-value --id "$reference_id")
    python3 "$PARSER" compare-stage "$VALUES" "$stage" "$raw"
}
post_stage() {
    sudo python3 "$PARSER" post-stage "$VALUES" "$1" \
        "$KBS_URL" "$TRUSTEE_PUBLIC_CERT" "$ADMIN_TOKEN"
}
STAGED=0
on_exit() {
    local status=$?
    if [[ $STAGED == 1 ]]; then
        if post_stage clear && verify_stage clear; then
            printf 'GPU driver approval was cleared after failure; VBIOS may have changed.\n' >&2
        else
            printf 'GPU approval state is UNKNOWN: compensating driver clear/readback failed. Stop GPU releases and investigate RVPS before proceeding.\n' >&2
        fi
    fi
    exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# RVPS processes named records separately. Clear and verify the driver gate
# before replacing VBIOS so a partial update cannot approve a mixed pair.
STAGED=1
post_stage clear
verify_stage clear
post_stage vbios
verify_stage vbios
post_stage drivers
verify_stage drivers
bash "$SCRIPT_DIR/10g-verify-gpu-reference-values.sh" "$VALUES"
[[ "$(sha256sum "$KBS_POLICY_DIR/resource-policy.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego")" == "$POLICY_BEFORE" ]] || \
    die 'An AS or KBS policy changed during GPU reference installation'
STAGED=0
printf 'Approved GPU driver/VBIOS references installed and read back. CPU references and AS/KBS policies were unchanged.\n'
