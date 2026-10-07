#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in ctr curl openssl systemctl; do need_cmd "${command}"; done
CA_SOURCE="${SCRIPT_DIR}/public/registry-ca.crt"
need_file "${CA_SOURCE}"
openssl x509 -in "${CA_SOURCE}" -noout -checkend 86400 >/dev/null || \
    die 'registry CA is invalid or expires within 24 hours'

CERT_DIR="/etc/containerd/certs.d/${REGISTRY_HOST}"
sudo install -d -m 0755 "${CERT_DIR}"
sudo install -o root -g root -m 0644 "${CA_SOURCE}" "${CERT_DIR}/ca.crt"
HOSTS="$(mktemp)"
trap 'rm -f "${HOSTS}"' EXIT
cat > "${HOSTS}" <<EOF
server = "https://${REGISTRY_HOST}"

[host."https://${REGISTRY_HOST}"]
  capabilities = ["pull", "resolve"]
  ca = "${CERT_DIR}/ca.crt"
EOF
sudo install -o root -g root -m 0644 "${HOSTS}" "${CERT_DIR}/hosts.toml"
sudo systemctl restart containerd
curl --fail --silent --show-error --cacert "${CA_SOURCE}" \
    --output /dev/null "https://${REGISTRY_HOST}/v2/"
sudo ctr --namespace k8s.io images pull --hosts-dir /etc/containerd/certs.d \
    "${REGISTRY_HOST}/service-verification/does-not-exist:expected" \
    >/dev/null 2>&1 && die 'unexpectedly pulled the deliberately nonexistent probe image'

printf 'Containerd trusts %s for anonymous pull/resolve only.\n' "${REGISTRY_HOST}"
printf 'CA SHA-256: '
sha256sum "${CA_SOURCE}" | cut -d' ' -f1
printf 'No registry publisher credential was installed.\n'
