#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

failures=0
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

configured_platform_values >/dev/null \
    && pass "approved SNP measurement allowlist and TCB floors are well formed" \
    || fail "invalid approved measurement allowlist or TCB floors"

for variable in \
    SNP_MIN_REPORTED_TCB_BOOTLOADER \
    SNP_MIN_REPORTED_TCB_TEE \
    SNP_MIN_REPORTED_TCB_SNP \
    SNP_MIN_REPORTED_TCB_MICROCODE; do
    if is_uint8 "${!variable-}"; then
        pass "${variable} is an explicit uint8 floor"
    else
        fail "${variable} must be independently approved and set to decimal 0..255"
    fi
done

[[ "$(uname -m)" == "x86_64" ]] && pass "x86_64 host" || fail "x86_64 is required"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] \
        && pass "Ubuntu 24.04" || fail "validated service OS is Ubuntu 24.04"
else
    fail "cannot identify operating system"
fi

if sudo -n true 2>/dev/null; then pass "passwordless sudo"; else fail "passwordless sudo is required"; fi
getent ahostsv4 "${SERVICE_FQDN}" >/dev/null \
    && pass "service FQDN resolves" || fail "service FQDN does not resolve"

free_gib="$(df -Pk "${HOME}" | awk 'NR == 2 {print int($4/1024/1024)}')"
((free_gib >= 40)) && pass "at least 40 GiB free" || fail "at least 40 GiB free is required"

for endpoint in \
    https://github.com/ \
    https://ghcr.io/v2/ \
    https://registry-1.docker.io/v2/ \
    https://static.crates.io/; do
    if curl --location --head --connect-timeout 8 --max-time 15 --silent \
        "${endpoint}" >/dev/null; then
        pass "reachable: ${endpoint}"
    else
        fail "unreachable: ${endpoint}"
    fi
done

if command -v ss >/dev/null 2>&1; then
    for port in 5000 8443; do
        if ss -ltnH "sport = :${port}" | grep -q .; then
            printf 'INFO  TCP %s is already in use; this is expected on an existing service\n' "${port}"
        else
            pass "TCP ${port} is available"
        fi
    done
fi

printf '\nPreflight failures: %s\n' "${failures}"
((failures == 0))
