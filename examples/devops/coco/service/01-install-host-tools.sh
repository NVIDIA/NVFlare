#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
require_sudo

[[ "$(uname -m)" == x86_64 ]] || die 'The pinned OPA binary requires an x86_64 service host'

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ca-certificates curl docker-compose-v2 docker.io git jq \
    nginx openssl pkg-config protobuf-compiler python3 python3-yaml uidmap xxd zstd
sudo systemctl enable --now docker nginx

# OPA is a required local checker, not a server. Fail closed if the pinned
# upstream release cannot be downloaded or its content differs.
OPA_VERSION=1.8.0
OPA_SHA256=1359b1bff233fc0a290066e864c75b8158e52756319757b6854df467fe7fc146
OPA_WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "${OPA_WORK_DIR}"' EXIT
curl --fail --location --silent --show-error --retry 3 \
    "https://github.com/open-policy-agent/opa/releases/download/v${OPA_VERSION}/opa_linux_amd64_static" \
    --output "${OPA_WORK_DIR}/opa"
printf '%s  %s\n' "${OPA_SHA256}" "${OPA_WORK_DIR}/opa" | sha256sum --check --strict
sudo install -o root -g root -m 0755 "${OPA_WORK_DIR}/opa" /usr/local/bin/opa
opa version

docker compose version
nginx -v
openssl version
python3 --version

printf 'Service host tools are installed.\n'
