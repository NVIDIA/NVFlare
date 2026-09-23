#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl nginx openssl python3; do need_cmd "${command}"; done

ROTATE=0
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && "$1" == "--rotate" ]] || die "usage: $0 [--rotate]"
    ROTATE=1
fi

# TRUSTEE_TLS_DIR is root-owned mode 0700 so the unprivileged service
# operator who runs this script cannot read it. A plain, unprivileged
# existence test on files inside it always reports "missing" for that
# operator, which used to make every rerun silently regenerate and overwrite
# an already-deployed certificate/key -- breaking trust for anything that
# already pinned the old certificate. Check with sudo, and only replace an
# existing certificate when the operator explicitly asks for rotation.
sudo install -d -m 0700 "${TRUSTEE_TLS_DIR}"
if [[ "${ROTATE}" -eq 1 ]]; then
    printf 'Rotating the Trustee TLS certificate as requested (--rotate).\n'
fi
if [[ "${ROTATE}" -eq 1 ]] \
    || ! sudo test -s "${TRUSTEE_TLS_DIR}/trustee.key" \
    || ! sudo test -s "${TRUSTEE_TLS_DIR}/trustee.crt"; then
    CONFIG="$(mktemp)"
    trap 'rm -f "${CONFIG}"' EXIT
    printf '%s\n' \
        '[req]' \
        'distinguished_name = dn' \
        'prompt = no' \
        'x509_extensions = extensions' \
        '[dn]' \
        "CN = ${SERVICE_FQDN}" \
        '[extensions]' \
        'basicConstraints = critical,CA:FALSE' \
        'keyUsage = critical,digitalSignature,keyEncipherment' \
        'extendedKeyUsage = serverAuth' \
        "subjectAltName = DNS:${SERVICE_FQDN}" > "${CONFIG}"
    sudo openssl req -new -newkey rsa:3072 -nodes -x509 -days 365 -sha256 \
        -config "${CONFIG}" \
        -keyout "${TRUSTEE_TLS_DIR}/trustee.key" \
        -out "${TRUSTEE_TLS_DIR}/trustee.crt"
else
    printf 'Existing Trustee TLS certificate found; leaving it in place. Pass --rotate to replace it.\n'
fi
sudo chown root:root "${TRUSTEE_TLS_DIR}/trustee.key" "${TRUSTEE_TLS_DIR}/trustee.crt"
sudo chmod 0600 "${TRUSTEE_TLS_DIR}/trustee.key"
sudo chmod 0644 "${TRUSTEE_TLS_DIR}/trustee.crt"

NGINX_CONFIG="$(mktemp)"
trap 'rm -f "${NGINX_CONFIG}"' EXIT
cat > "${NGINX_CONFIG}" <<EOF
server {
    listen 8443 ssl;
    server_name ${SERVICE_FQDN};

    ssl_certificate ${TRUSTEE_TLS_DIR}/trustee.crt;
    ssl_certificate_key ${TRUSTEE_TLS_DIR}/trustee.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_tickets off;
    large_client_header_buffers 4 128k;
    client_max_body_size 16m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 300s;
    }
}
EOF
sudo install -m 0644 "${NGINX_CONFIG}" /etc/nginx/sites-available/trustee.conf
sudo ln -sfn /etc/nginx/sites-available/trustee.conf /etc/nginx/sites-enabled/trustee.conf
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx

sudo install -o "$(id -u)" -g "$(id -g)" -m 0644 \
    "${TRUSTEE_TLS_DIR}/trustee.crt" "${TRUSTEE_PUBLIC_CERT}"
wait_https "${KBS_URL}/healthz" "${TRUSTEE_PUBLIC_CERT}" 90

DUMMY_HEADER="$(mktemp)"
trap 'rm -f "${NGINX_CONFIG}" "${DUMMY_HEADER}"' EXIT
python3 - "${DUMMY_HEADER}" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).write_text("Authorization: Bearer " + "A" * 40000 + "\n")
PY
curl --http1.1 --fail --silent --show-error \
    --cacert "${TRUSTEE_PUBLIC_CERT}" --header "@${DUMMY_HEADER}" \
    --output /dev/null "${KBS_URL}/healthz"

printf 'Trustee TLS endpoint is ready: %s\n' "${KBS_URL}"
printf 'Public certificate: %s\n' "${TRUSTEE_PUBLIC_CERT}"
printf 'Certificate SHA-256: '
sha256sum "${TRUSTEE_PUBLIC_CERT}" | cut -d' ' -f1
printf 'Normal and 40,000-byte-header health probes passed.\n'
