#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/platform.sh
source "${SCRIPT_DIR}/lib/platform.sh"

need_file "${PUBLIC_DIR}/trustee.crt"
need_file "${PUBLIC_DIR}/registry-ca.crt"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl docker.io golang-go jq openssl python3-yaml skopeo zstd

sudo install -m 0644 "${PUBLIC_DIR}/registry-ca.crt" \
    /usr/local/share/ca-certificates/coco-registry-ca.crt
sudo update-ca-certificates
sudo install -d -m 0755 "/etc/docker/certs.d/${REGISTRY_HOST}:${REGISTRY_PORT}"
sudo install -m 0644 "${PUBLIC_DIR}/registry-ca.crt" \
    "/etc/docker/certs.d/${REGISTRY_HOST}:${REGISTRY_PORT}/ca.crt"
sudo systemctl enable --now docker

COSIGN_PATH="${BIN_DIR}/cosign-${COSIGN_VERSION}"
if [[ ! -x "${COSIGN_PATH}" ]]; then
    curl --fail --location --proto '=https' --tlsv1.2 \
        --output "${COSIGN_PATH}.download" "${COSIGN_URL}"
    sha256_check "${COSIGN_SHA256}" "${COSIGN_PATH}.download"
    install -m 0755 "${COSIGN_PATH}.download" "${COSIGN_PATH}"
    rm -f "${COSIGN_PATH}.download"
fi
ln -sfn "${COSIGN_PATH}" "${BIN_DIR}/cosign"

KATA_ARCHIVE="${TOOLS_DIR}/kata-tools-static-${KATA_VERSION}-amd64.tar.zst"
if [[ ! -s "${KATA_ARCHIVE}" ]]; then
    curl --fail --location --proto '=https' --tlsv1.2 \
        --output "${KATA_ARCHIVE}.download" "${KATA_TOOLS_URL}"
    sha256_check "${KATA_TOOLS_SHA256}" "${KATA_ARCHIVE}.download"
    mv "${KATA_ARCHIVE}.download" "${KATA_ARCHIVE}"
fi
sha256_check "${KATA_TOOLS_SHA256}" "${KATA_ARCHIVE}"

if [[ ! -x "${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/bin/genpolicy" ]]; then
    mkdir -p "${TOOLS_DIR}/kata-${KATA_VERSION}"
    tar --use-compress-program=unzstd -xf "${KATA_ARCHIVE}" \
        -C "${TOOLS_DIR}/kata-${KATA_VERSION}"
fi

need_file "${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/bin/genpolicy"
need_file "${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/share/defaults/kata-containers/rules.rego"
need_file "${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/share/defaults/kata-containers/genpolicy-settings.json"

"${BIN_DIR}/cosign" version
"${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/bin/genpolicy" \
    -j "${TOOLS_DIR}/kata-${KATA_VERSION}/opt/kata/share/defaults/kata-containers/genpolicy-settings.json" --version
sudo docker version --format 'docker client={{.Client.Version}} server={{.Server.Version}}'
skopeo --version

curl --silent --show-error --cacert "${PUBLIC_DIR}/trustee.crt" \
    "${KBS_URL}/kbs/v0/" >/dev/null

# The service challenges every request from the publisher's exact source /32
# so Docker and Skopeo learn that credentials are required before a mutation.
# At this pre-credential stage, require that challenge instead of accepting an
# anonymous 200 response.
REGISTRY_HEADERS="$(mktemp)"
trap 'rm -f "${REGISTRY_HEADERS}"' EXIT
REGISTRY_STATUS="$(curl --silent --show-error \
    --cacert "${PUBLIC_DIR}/registry-ca.crt" \
    --dump-header "${REGISTRY_HEADERS}" --output /dev/null \
    --write-out '%{http_code}' \
    "https://${REGISTRY_HOST}:${REGISTRY_PORT}/v2/")"
[[ "${REGISTRY_STATUS}" == 401 ]] || \
    die "expected registry publisher challenge, received HTTP ${REGISTRY_STATUS}"
grep -Eiq '^www-authenticate:[[:space:]]*Basic realm="CoCo registry publisher"' \
    "${REGISTRY_HEADERS}" || die 'registry publisher challenge is missing or unexpected'

printf 'Owner tools and public trust anchors are ready.\n'
printf 'Registry publisher-origin authentication challenge verified.\n'
