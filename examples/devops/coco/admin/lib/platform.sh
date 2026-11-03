#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common-base.sh"
set -Eeuo pipefail
umask 077

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -s "${SCRIPT_ROOT}/platform.env" ]] || {
    echo 'Copy platform.env.example to platform.env and review the existing tool and launch-profile paths.' >&2
    exit 2
}
source "${SCRIPT_ROOT}/platform.env"
source "${SCRIPT_ROOT}/lib/validate-config.sh"
validate_private_root "${WORK_ROOT}"
for variable in GENPOLICY RULES SETTINGS WORKLOAD_LAUNCH_PROFILE; do
    [[ ${!variable:-} == /* ]] || config_error "Set ${variable} to an existing absolute path."
done

PUBLIC_DIR="${SCRIPT_ROOT}/public"
SECRETS_DIR="${WORK_ROOT}/secrets"
RELEASES_DIR="${WORK_ROOT}/releases"
SIGNING_DIR="${SECRETS_DIR}/signing"
mkdir -p "${SECRETS_DIR}" "${RELEASES_DIR}" "${SIGNING_DIR}"
chmod 0700 "${SECRETS_DIR}" "${RELEASES_DIR}" "${SIGNING_DIR}"

kbs_uri() { printf 'kbs:///%s' "$1"; }
