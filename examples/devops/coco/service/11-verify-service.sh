#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl docker nginx openssl ss; do need_cmd "${command}"; done
need_file "${TRUSTEE_PUBLIC_CERT}"
need_file "${PUBLISHER_DIR}/registry-ca.crt"
need_file "${KBS_CLIENT}"
need_file "${ADMIN_TOKEN}"
need_file "${SCRIPT_DIR}/policies/default_cpu.rego"
need_file "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego"

PLATFORM_VALUES_JSON="$(configured_platform_values)"
VALUES_FILE="$(mktemp)"
trap 'rm -f -- "$VALUES_FILE"' EXIT
printf '%s\n' "$PLATFORM_VALUES_JSON" > "$VALUES_FILE"
require_uint8 SNP_MIN_REPORTED_TCB_BOOTLOADER "${SNP_MIN_REPORTED_TCB_BOOTLOADER-}"
require_uint8 SNP_MIN_REPORTED_TCB_TEE "${SNP_MIN_REPORTED_TCB_TEE-}"
require_uint8 SNP_MIN_REPORTED_TCB_SNP "${SNP_MIN_REPORTED_TCB_SNP-}"
require_uint8 SNP_MIN_REPORTED_TCB_MICROCODE "${SNP_MIN_REPORTED_TCB_MICROCODE-}"

cmp --silent "${SCRIPT_DIR}/policies/default_cpu.rego" \
    "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego" \
    || die "active CPU policy differs from the reviewed TCB-floor policy"

bash "$SCRIPT_DIR/10-verify-platform-reference-values.sh" "$VALUES_FILE"

curl --fail --silent --show-error --cacert "${TRUSTEE_PUBLIC_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"
curl --fail --silent --show-error --cacert "${PUBLISHER_DIR}/registry-ca.crt" \
    --output /dev/null "https://${SERVICE_FQDN}:${REGISTRY_PORT}/v2/"
sudo nginx -t
sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${TRUSTEE_COMPOSE}" ps
sudo docker compose -p coco-registry -f "${REGISTRY_ROOT}/docker-compose.yml" ps

LISTENERS="$(sudo ss -lntH)"
for port in 8080 50003 50004 "${REGISTRY_BACKEND_PORT}"; do
    if grep -Eq "(0\.0\.0\.0|\[::\]):${port}([[:space:]]|$)" <<<"${LISTENERS}"; then
        die "backend port ${port} is exposed beyond loopback"
    fi
done
grep -Eq "127\.0\.0\.1:8080([[:space:]]|$)" <<<"${LISTENERS}" || \
    die 'KBS backend is not listening on loopback port 8080'
grep -Eq "127\.0\.0\.1:${REGISTRY_BACKEND_PORT}([[:space:]]|$)" <<<"${LISTENERS}" || \
    die 'registry backend is not listening on loopback'

[[ "$(sudo stat -c '%U:%G %a' "${ADMIN_TOKEN}")" == 'root:root 600' ]]
[[ "$(sudo stat -c '%U:%G %a' "${TRUSTEE_TLS_DIR}/trustee.key")" == 'root:root 600' ]]
[[ "$(sudo stat -c '%U:%G %a' "${REGISTRY_ROOT}/pki/ca.key")" == 'root:root 600' ]]
[[ "$(sudo stat -c '%U:%G %a' "${REGISTRY_ROOT}/tls/server.key")" == 'root:root 600' ]]

openssl verify -CAfile "${PUBLISHER_DIR}/registry-ca.crt" \
    <(sudo cat "${REGISTRY_ROOT}/tls/server.crt")
printf 'Service verification passed. Public endpoints: 8443 (Trustee), 5000 (registry).\n'
printf 'Backend ports 8080, 50003, 50004, and %s are loopback-only.\n' \
    "${REGISTRY_BACKEND_PORT}"
printf 'SNP launch measurement and all four minimum reported TCB floors match platform.env.\n'
