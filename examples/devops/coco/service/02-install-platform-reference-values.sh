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
if [[ -n ${PLATFORM_REFERENCE_VALUES_FILE:-} ]]; then
    # Preserve the previous snapshot verbatim, including retired reference schemas.
    [[ -f "$PLATFORM_REFERENCE_VALUES_FILE" && -r "$PLATFORM_REFERENCE_VALUES_FILE" ]] || \
        die 'Previous reference snapshot must be a readable regular file'
    install -m 0600 "$PLATFORM_REFERENCE_VALUES_FILE" "$BACKUP.references.json"
    printf 'Previous reference JSON snapshot: %s\n' "$BACKUP.references.json"
fi
python3 "$PARSER" update-env "$VALUES" "$SCRIPT_DIR/platform.env"
printf 'Saved approved reference snapshot and updated platform.env. Previous configuration: %s\n' "$BACKUP"
if [[ ${3:-} == --configure-only ]]; then
    printf 'Configuration-only: no RVPS, AS or KBS setting was changed.\n'
    exit 0
fi
source "$SCRIPT_DIR/lib/common.sh"
trap 'printf "Installation failed: references may be partially staged. Do not launch workloads; correct the failure and rerun this command. No automatic rollback is performed.\n" >&2' ERR
install_platform_references "$VALUES"
bash "$SCRIPT_DIR/10-verify-platform-reference-values.sh" "$VALUES"
[[ "$(sha256sum "$KBS_POLICY_DIR/resource-policy.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_cpu.rego" \
    "$AS_STORAGE_DIR/attestation_service_policy/default_gpu.rego")" == "$POLICY_BEFORE" ]] || \
    die 'An AS or KBS policy changed during reference installation; investigate before proceeding.'
printf 'Installed approved RVPS references for the selected TEE. Other TEE references are unchanged.\n'
printf 'AS CPU/GPU and KBS release-policy files are unchanged.\n'
