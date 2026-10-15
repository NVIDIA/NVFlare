#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common-base.sh"

set -Eeuo pipefail
umask 077

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../platform.env
[[ -s "${SCRIPT_ROOT}/platform.env" ]] || { echo 'Copy platform.env.example to platform.env, chmod 600, and review it first.' >&2; exit 2; }
source "${SCRIPT_ROOT}/platform.env"
source "${SCRIPT_ROOT}/lib/validate-config.sh"
validate_target_host
validate_private_root "$WORK_ROOT"

PUBLIC_DIR="${SCRIPT_ROOT}/public"
SECRETS_DIR="${WORK_ROOT}/secrets"
RELEASES_DIR="${WORK_ROOT}/releases"
TOOLS_DIR="${WORK_ROOT}/tools"
BIN_DIR="${HOME}/.local/bin"
SIGNING_DIR="${SECRETS_DIR}/signing"

export PATH="${BIN_DIR}:${PATH}"

mkdir -p "${PUBLIC_DIR}" "${SECRETS_DIR}" "${RELEASES_DIR}" \
    "${TOOLS_DIR}" "${BIN_DIR}" "${SIGNING_DIR}"
chmod 0700 "${SECRETS_DIR}" "${RELEASES_DIR}" "${SIGNING_DIR}"




sha256_check() {
    local expected="$1"
    local file="$2"
    printf '%s  %s\n' "${expected}" "${file}" | sha256sum --check --status \
        || die "SHA-256 mismatch: ${file}"
}

kbs_uri() {
    printf 'kbs:///%s' "$1"
}
