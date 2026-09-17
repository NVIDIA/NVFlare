#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_CONTEXT="$REPO_ROOT"
TEMP_CHECKOUT=""

cleanup() {
    [[ -z "$TEMP_CHECKOUT" ]] || rm -rf "$TEMP_CHECKOUT"
}
trap cleanup EXIT

if [[ ! -d "$BUILD_CONTEXT/nvflare" || ! -f "$BUILD_CONTEXT/setup.py" ]]; then
    command -v git >/dev/null 2>&1 || {
        echo "git is required to prepare the revision-matched Docker build context." >&2
        exit 1
    }
    REVISION="$(nvflare examples revision --dir "$REPO_ROOT")"
    TEMP_CHECKOUT="$(mktemp -d "${TMPDIR:-/tmp}/nvflare-docker.XXXXXX")"
    mkdir "$TEMP_CHECKOUT/repository" "$TEMP_CHECKOUT/source"
    git -C "$TEMP_CHECKOUT/repository" init --quiet
    git -C "$TEMP_CHECKOUT/repository" remote add origin https://github.com/NVIDIA/NVFlare.git
    git -C "$TEMP_CHECKOUT/repository" fetch --quiet --depth=1 origin "$REVISION"
    git -C "$TEMP_CHECKOUT/repository" archive FETCH_HEAD | tar -x -C "$TEMP_CHECKOUT/source"
    BUILD_CONTEXT="$TEMP_CHECKOUT/source"
fi

if [[ -z "${NVFL_BASE_VERSION:-}" ]]; then
    VERSION_TAG="$(git -C "$BUILD_CONTEXT" describe --tags --abbrev=0 --match '[0-9]*' 2>/dev/null || true)"
    if [[ "$VERSION_TAG" =~ ^([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        NVFL_BASE_VERSION="${BASH_REMATCH[1]}"
    else
        NVFL_BASE_VERSION="$(python3 -c 'import re, nvflare; match = re.match(r"[0-9]+\.[0-9]+\.[0-9]+", nvflare.__version__); print(match.group(0) if match else "")')"
        if [[ -z "$NVFL_BASE_VERSION" ]]; then
            echo "Unable to determine NVFL_BASE_VERSION from Git metadata or the installed NVFlare package." >&2
            echo "Set NVFL_BASE_VERSION explicitly and retry." >&2
            exit 1
        fi
    fi
fi

echo "Building NVFlare Docker images with NVFL_BASE_VERSION=$NVFL_BASE_VERSION"

docker build \
    --build-arg NVFL_BASE_VERSION="$NVFL_BASE_VERSION" \
    -t nvflare-site:latest \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$BUILD_CONTEXT"
docker build \
    --build-arg NVFL_BASE_VERSION="$NVFL_BASE_VERSION" \
    -t nvflare-job:latest \
    -f "$SCRIPT_DIR/Dockerfile.nvflare-job" \
    "$BUILD_CONTEXT"
