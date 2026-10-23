#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v go >/dev/null 2>&1 || {
    printf 'ERROR: Go is not installed; run ../00-install-tools.sh first.\n' >&2
    exit 1
}

(
    cd "${SCRIPT_DIR}"
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
        go build -buildvcs=false -trimpath -ldflags='-s -w' \
        -o coco-app main.go
)
chmod 0555 "${SCRIPT_DIR}/coco-app"

printf 'Built static example workload: %s\n' "${SCRIPT_DIR}/coco-app"
sha256sum "${SCRIPT_DIR}/main.go" "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}/coco-app"
