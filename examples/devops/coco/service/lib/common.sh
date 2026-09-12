#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../platform.env
[[ -s "${SCRIPT_ROOT}/platform.env" ]] || { echo 'Copy platform.env.example to platform.env, chmod 600, and review it first.' >&2; exit 2; }
source "${SCRIPT_ROOT}/platform.env"
source "${SCRIPT_ROOT}/lib/validate-config.sh"
validate_target_host
validate_service_host "$SERVICE_FQDN"
validate_private_root "$TRUSTEE_ROOT"
validate_private_root "$PUBLISHER_DIR"
[[ $REGISTRY_PORT == 5000 && $REGISTRY_BACKEND_PORT == 5001 && $KBS_URL == "https://${SERVICE_FQDN}:8443" ]] || config_error 'Use registry TLS 5000, loopback backend 5001, and KBS HTTPS 8443.'

TRUSTEE_COMPOSE="${TRUSTEE_ROOT}/docker-compose.yml"
KBS_CLIENT="${TRUSTEE_ROOT}/kbs-client-snp-tdx-${TRUSTEE_LABEL}"
ADMIN_TOKEN="${TRUSTEE_ROOT}/kbs/config/docker-compose/admin-token"
KBS_STORAGE_DIR="${TRUSTEE_ROOT}/kbs/data/kbs-storage"
KBS_POLICY_DIR="${TRUSTEE_ROOT}/kbs/data/kbs-policy"
AS_STORAGE_DIR="${TRUSTEE_ROOT}/kbs/data/attestation-service"
REFERENCE_DIR="${TRUSTEE_ROOT}/kbs/data/reference-values"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

need_file() {
    [[ -s "$1" ]] || die "required file is missing or empty: $1"
}

is_uint8() {
    local value="${1-}"
    [[ "${value}" =~ ^(0|[1-9][0-9]{0,2})$ ]] || return 1
    (( 10#${value} <= 255 ))
}

require_uint8() {
    local name="$1" value="${2-}"
    is_uint8 "${value}" \
        || die "${name} must be an explicitly approved decimal integer from 0 through 255"
}

read_integer_reference() {
    local reference_id="$1" raw cleaned
    raw="$(kbs_admin get-reference-value --id "${reference_id}")"
    cleaned="$(printf '%s' "${raw}" | tr -d '[:space:]\"')"
    is_uint8 "${cleaned}" \
        || die "RVPS reference ${reference_id} is missing or is not one scalar uint8"
    printf '%s\n' "${cleaned}"
}

verify_integer_reference() {
    local reference_id="$1" expected="$2" actual
    actual="$(read_integer_reference "${reference_id}")"
    [[ "${actual}" == "${expected}" ]] \
        || die "RVPS reference ${reference_id} is ${actual}, expected ${expected}"
}

require_sudo() {
    need_cmd sudo
    sudo -n true || die "passwordless non-interactive sudo is required"
}

wait_https() {
    local url="$1" cert="$2" attempts="${3:-90}"
    local index
    for ((index = 1; index <= attempts; index++)); do
        if curl --fail --silent --show-error --cacert "${cert}" \
            --output /dev/null "${url}"; then
            return 0
        fi
        sleep 1
    done
    die "HTTPS endpoint did not become ready: ${url}"
}

kbs_admin() {
    sudo "${KBS_CLIENT}" --url "${KBS_URL}" --cert-file "${TRUSTEE_PUBLIC_CERT}" \
        config --admin-token-file "${ADMIN_TOKEN}" "$@"
}

# Shared by initial policy installation and subsequent reference updates.
lock_platform_reference_update() {
    need_cmd flock
    exec {PLATFORM_REFERENCE_LOCK_FD}>"${SCRIPT_ROOT}/.platform-reference-update.lock"
    flock -n "$PLATFORM_REFERENCE_LOCK_FD" || die 'Another platform-reference update is in progress'
}

configured_platform_values() {
    python3 "$SCRIPT_ROOT/lib/platform-reference-values.py" from-env \
        "${SNP_LAUNCH_MEASUREMENT:-}" "${SNP_MIN_REPORTED_TCB_BOOTLOADER:-}" \
        "${SNP_MIN_REPORTED_TCB_TEE:-}" "${SNP_MIN_REPORTED_TCB_SNP:-}" \
        "${SNP_MIN_REPORTED_TCB_MICROCODE:-}"
}

install_measurement_allowlist() {
    sudo python3 "$SCRIPT_ROOT/lib/platform-reference-values.py" install-measurements \
        "$1" "$KBS_URL" "$TRUSTEE_PUBLIC_CERT" "$ADMIN_TOKEN"
}
