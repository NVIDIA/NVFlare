#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl docker nginx openssl python3; do need_cmd "${command}"; done

python3 - "${REGISTRY_PUBLISHER_CIDR}" <<'PY'
import ipaddress
import sys

network = ipaddress.ip_network(sys.argv[1], strict=True)
if network.version != 4 or network.prefixlen != 32:
    raise SystemExit("REGISTRY_PUBLISHER_CIDR must be one exact IPv4 /32")
if any(network.subnet_of(ipaddress.ip_network(block)) for block in
       ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")):
    raise SystemExit("Replace the documentation publisher address with the actual authenticated egress IPv4/32")
PY

sudo install -d -m 0700 "${REGISTRY_ROOT}/pki" "${REGISTRY_ROOT}/tls" \
    "${REGISTRY_ROOT}/auth"
sudo install -d -m 0750 "${REGISTRY_DATA}"

CA_CREATED=false
if ! sudo test -s "${REGISTRY_ROOT}/pki/ca.key" || \
   ! sudo test -s "${REGISTRY_ROOT}/pki/ca.crt"; then
    sudo openssl req -new -newkey rsa:4096 -nodes -x509 -days 3650 -sha256 \
        -subj "/CN=CoCo private registry CA" \
        -keyout "${REGISTRY_ROOT}/pki/ca.key" \
        -out "${REGISTRY_ROOT}/pki/ca.crt"
    CA_CREATED=true
fi

if [[ "${CA_CREATED}" == true ]] || \
   ! sudo test -s "${REGISTRY_ROOT}/tls/server.key" || \
   ! sudo test -s "${REGISTRY_ROOT}/tls/server.crt"; then
    CSR="$(sudo mktemp "${REGISTRY_ROOT}/.server-csr.XXXXXX")"
    EXT="$(sudo mktemp "${REGISTRY_ROOT}/.server-ext.XXXXXX")"
    trap 'sudo rm -f "${CSR}" "${EXT}"' EXIT
    sudo openssl req -new -newkey rsa:3072 -nodes -sha256 \
        -subj "/CN=${SERVICE_FQDN}" \
        -keyout "${REGISTRY_ROOT}/tls/server.key" -out "${CSR}"
    printf '%s\n' \
        'basicConstraints=critical,CA:FALSE' \
        'keyUsage=critical,digitalSignature,keyEncipherment' \
        'extendedKeyUsage=serverAuth' \
        "subjectAltName=DNS:${SERVICE_FQDN}" | sudo tee "${EXT}" >/dev/null
    sudo openssl x509 -req -days 365 -sha256 -in "${CSR}" \
        -CA "${REGISTRY_ROOT}/pki/ca.crt" \
        -CAkey "${REGISTRY_ROOT}/pki/ca.key" -CAcreateserial \
        -extfile "${EXT}" -out "${REGISTRY_ROOT}/tls/server.crt"
    sudo rm -f "${CSR}" "${EXT}"
    trap - EXIT
fi

sudo chown -R root:root "${REGISTRY_ROOT}"
sudo chmod 0600 "${REGISTRY_ROOT}/pki/ca.key" "${REGISTRY_ROOT}/tls/server.key"
sudo chmod 0644 "${REGISTRY_ROOT}/pki/ca.crt" "${REGISTRY_ROOT}/tls/server.crt"

install -d -m 0700 "${PUBLISHER_DIR}"
if [[ ! -s "${PUBLISHER_DIR}/password" ]]; then
    openssl rand -base64 36 > "${PUBLISHER_DIR}/password"
fi
printf '%s\n' "${REGISTRY_PUBLISHER}" > "${PUBLISHER_DIR}/username"
chmod 0600 "${PUBLISHER_DIR}/username" "${PUBLISHER_DIR}/password"
sudo install -o "$(id -u)" -g "$(id -g)" -m 0644 \
    "${REGISTRY_ROOT}/pki/ca.crt" "${PUBLISHER_DIR}/registry-ca.crt"

PASSWORD_HASH="$(openssl passwd -apr1 -in "${PUBLISHER_DIR}/password")"
printf '%s:%s\n' "${REGISTRY_PUBLISHER}" "${PASSWORD_HASH}" | \
    sudo tee "${REGISTRY_ROOT}/auth/htpasswd" >/dev/null
sudo chown root:www-data "${REGISTRY_ROOT}/auth" \
    "${REGISTRY_ROOT}/auth/htpasswd"
sudo chmod 0750 "${REGISTRY_ROOT}/auth"
sudo chmod 0640 "${REGISTRY_ROOT}/auth/htpasswd"

COMPOSE="$(mktemp)"
trap 'rm -f "${COMPOSE}"' EXIT
cat > "${COMPOSE}" <<EOF
services:
  registry:
    image: ${REGISTRY_IMAGE}
    restart: unless-stopped
    ports:
      - "127.0.0.1:${REGISTRY_BACKEND_PORT}:5000"
    environment:
      REGISTRY_HTTP_ADDR: 0.0.0.0:5000
      REGISTRY_HTTP_TLS_CERTIFICATE: /tls/server.crt
      REGISTRY_HTTP_TLS_KEY: /tls/server.key
    volumes:
      - ${REGISTRY_DATA}:/var/lib/registry
      - ${REGISTRY_ROOT}/tls:/tls:ro
EOF
sudo install -m 0644 "${COMPOSE}" "${REGISTRY_ROOT}/docker-compose.yml"
(
    cd "${REGISTRY_ROOT}"
    sudo docker compose -p coco-registry up -d --pull always --force-recreate
)

NGINX_CONFIG="$(mktemp)"
PUBLISHER_ORIGIN_CONFIG="$(mktemp)"
trap 'rm -f "${COMPOSE}" "${NGINX_CONFIG}" "${PUBLISHER_ORIGIN_CONFIG}"' EXIT
cat > "${PUBLISHER_ORIGIN_CONFIG}" <<EOF
geo \$coco_registry_from_publisher_origin {
    default 0;
    ${REGISTRY_PUBLISHER_CIDR} 1;
}

map "\$coco_registry_from_publisher_origin:\$request_method" \$coco_registry_auth_realm {
    default "CoCo registry publisher";
    ~^0:(GET|HEAD)$ off;
}
EOF
sudo install -m 0644 "${PUBLISHER_ORIGIN_CONFIG}" \
    /etc/nginx/conf.d/coco-registry-publisher-origin.conf
cat > "${NGINX_CONFIG}" <<EOF
server {
    listen ${REGISTRY_PORT} ssl;
    server_name ${SERVICE_FQDN};

    ssl_certificate ${REGISTRY_ROOT}/tls/server.crt;
    ssl_certificate_key ${REGISTRY_ROOT}/tls/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_tickets off;
    client_max_body_size 0;

    location /v2/ {
        auth_basic \$coco_registry_auth_realm;
        auth_basic_user_file ${REGISTRY_ROOT}/auth/htpasswd;
        proxy_pass https://127.0.0.1:${REGISTRY_BACKEND_PORT};
        proxy_ssl_verify on;
        proxy_ssl_trusted_certificate ${REGISTRY_ROOT}/pki/ca.crt;
        proxy_ssl_name ${SERVICE_FQDN};
        proxy_ssl_server_name on;
        # Distribution derives resumable-upload Location URLs from these
        # forwarded values. Preserve :5000 or clients would follow port 443.
        proxy_set_header Host \$http_host;
        proxy_set_header X-Forwarded-Host \$http_host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Docker-Distribution-Api-Version registry/2.0;
        proxy_read_timeout 900s;
    }
}
EOF
sudo install -m 0644 "${NGINX_CONFIG}" /etc/nginx/sites-available/coco-registry.conf
sudo ln -sfn /etc/nginx/sites-available/coco-registry.conf \
    /etc/nginx/sites-enabled/coco-registry.conf
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx

REGISTRY_URL="https://${SERVICE_FQDN}:${REGISTRY_PORT}/v2/"
for _ in $(seq 1 60); do
    if curl --fail --silent --show-error --cacert "${PUBLISHER_DIR}/registry-ca.crt" \
        --output /dev/null "${REGISTRY_URL}"; then break; fi
    sleep 1
done
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --cacert "${PUBLISHER_DIR}/registry-ca.crt" -X POST \
    "${REGISTRY_URL}service-verification/blobs/uploads/")" == "401" ]]
NETRC="$(mktemp)"
UPLOAD_HEADERS="$(mktemp)"
trap 'rm -f "${COMPOSE}" "${NGINX_CONFIG}" "${PUBLISHER_ORIGIN_CONFIG}" "${NETRC}" "${UPLOAD_HEADERS}"' EXIT
printf 'machine %s login %s password %s\n' \
    "${SERVICE_FQDN}" "${REGISTRY_PUBLISHER}" \
    "$(<"${PUBLISHER_DIR}/password")" > "${NETRC}"
chmod 0600 "${NETRC}"
[[ "$(curl --silent --dump-header "${UPLOAD_HEADERS}" \
    --output /dev/null --write-out '%{http_code}' \
    --cacert "${PUBLISHER_DIR}/registry-ca.crt" \
    --netrc-file "${NETRC}" \
    -X POST "${REGISTRY_URL}service-verification/blobs/uploads/")" == "202" ]]
UPLOAD_LOCATION="$(awk 'tolower($1) == "location:" {sub(/^[^:]*:[[:space:]]*/, ""); sub(/\r$/, ""); print; exit}' "${UPLOAD_HEADERS}")"
[[ "${UPLOAD_LOCATION}" == "https://${SERVICE_FQDN}:${REGISTRY_PORT}/v2/"* ]] \
    || die "registry upload Location lost the external TLS host/port: ${UPLOAD_LOCATION}"
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --cacert "${PUBLISHER_DIR}/registry-ca.crt" \
    --netrc-file "${NETRC}" -X DELETE "${UPLOAD_LOCATION}")" == "204" ]]

printf 'Private registry is ready: %s:%s\n' "${SERVICE_FQDN}" "${REGISTRY_PORT}"
printf 'Anonymous pulls are enabled; all mutations require publisher credentials.\n'
printf 'Publisher clients from %s are challenged on /v2/ so OCI clients authenticate.\n' \
    "${REGISTRY_PUBLISHER_CIDR}"
printf 'Publisher bundle (service/admin only): %s\n' "${PUBLISHER_DIR}"
printf 'Registry CA SHA-256: '
sha256sum "${PUBLISHER_DIR}/registry-ca.crt" | cut -d' ' -f1
printf 'Do not copy the password, CA key, or server key to coco.\n'
