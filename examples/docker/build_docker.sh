#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${NVFL_BASE_VERSION:-}" ]]; then
    VERSION_TAG="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 --match '[0-9]*' 2>/dev/null || true)"
    if [[ "$VERSION_TAG" =~ ^([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        NVFL_BASE_VERSION="${BASH_REMATCH[1]}"
    else
        echo "Unable to determine NVFL_BASE_VERSION; set it explicitly before building." >&2
        exit 1
    fi
fi

echo "Building NVFlare Docker images with NVFL_BASE_VERSION=$NVFL_BASE_VERSION"

docker build \
    --build-arg NVFL_BASE_VERSION="$NVFL_BASE_VERSION" \
    -t nvflare-site:latest \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$REPO_ROOT"
docker build \
    --build-arg NVFL_BASE_VERSION="$NVFL_BASE_VERSION" \
    -t nvflare-job:latest \
    -f "$SCRIPT_DIR/Dockerfile.nvflare-job" \
    "$REPO_ROOT"
