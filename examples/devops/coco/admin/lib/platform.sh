#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../platform.env
[[ -s "${SCRIPT_ROOT}/platform.env" ]] || { echo 'Copy platform.env.example to platform.env, chmod 600, and review it first.' >&2; exit 2; }
source "${SCRIPT_ROOT}/platform.env"
source "${SCRIPT_ROOT}/lib/validate-config.sh"
validate_target_host
validate_service_host "$REGISTRY_HOST"
validate_private_root "$WORK_ROOT"
[[ $REGISTRY_PORT == 5000 && $KBS_URL == "https://${REGISTRY_HOST}:8443" ]] || config_error 'Use registry TLS port 5000 and KBS HTTPS port 8443 on the same secure-services DNS name.'

PUBLIC_DIR="${SCRIPT_ROOT}/public"
SECRETS_DIR="${WORK_ROOT}/secrets"
RELEASES_DIR="${WORK_ROOT}/releases"
TOOLS_DIR="${WORK_ROOT}/tools"
BIN_DIR="${HOME}/.local/bin"
SIGNING_DIR="${SECRETS_DIR}/signing"
REGISTRY_SECRET_DIR="${SECRETS_DIR}/registry"

export PATH="${BIN_DIR}:${PATH}"

mkdir -p "${PUBLIC_DIR}" "${SECRETS_DIR}" "${RELEASES_DIR}" \
    "${TOOLS_DIR}" "${BIN_DIR}" "${SIGNING_DIR}" "${REGISTRY_SECRET_DIR}"
chmod 0700 "${SECRETS_DIR}" "${RELEASES_DIR}" "${SIGNING_DIR}" \
    "${REGISTRY_SECRET_DIR}"

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

sha256_check() {
    local expected="$1"
    local file="$2"
    printf '%s  %s\n' "${expected}" "${file}" | sha256sum --check --status \
        || die "SHA-256 mismatch: ${file}"
}

registry_base() {
    printf '%s:%s' "${REGISTRY_HOST}" "${REGISTRY_PORT}"
}

kbs_uri() {
    printf 'kbs:///%s' "$1"
}
