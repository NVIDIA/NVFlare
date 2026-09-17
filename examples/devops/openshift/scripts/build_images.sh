#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_CONTEXT="$DOWNLOAD_ROOT"
TEMP_SOURCE=""
CONTAINER_TOOL="${CONTAINER_TOOL:-podman}"
PARENT_IMAGE="${PARENT_IMAGE:?Set PARENT_IMAGE to the parent image tag to build and push.}"
WORKLOAD_IMAGE="${WORKLOAD_IMAGE:?Set WORKLOAD_IMAGE to the workload image tag to build and push.}"

cleanup() {
  [[ -z "$TEMP_SOURCE" ]] || rm -rf "$TEMP_SOURCE"
}
trap cleanup EXIT

command -v "$CONTAINER_TOOL" >/dev/null 2>&1 || {
  echo "Required container tool not found: $CONTAINER_TOOL" >&2
  exit 1
}

if [[ ! -d "$BUILD_CONTEXT/nvflare" || ! -f "$BUILD_CONTEXT/docker/Dockerfile.parent" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git is required to prepare the revision-matched image build context." >&2
    exit 1
  }
  REVISION="$(nvflare examples revision --dir "$DOWNLOAD_ROOT")"
  TEMP_SOURCE="$(mktemp -d "${TMPDIR:-/tmp}/nvflare-openshift.XXXXXX")"
  mkdir "$TEMP_SOURCE/repository" "$TEMP_SOURCE/source"
  git -C "$TEMP_SOURCE/repository" init --quiet
  git -C "$TEMP_SOURCE/repository" remote add origin https://github.com/NVIDIA/NVFlare.git
  git -C "$TEMP_SOURCE/repository" fetch --quiet --depth=1 origin "$REVISION"
  git -C "$TEMP_SOURCE/repository" archive FETCH_HEAD | tar -x -C "$TEMP_SOURCE/source"
  BUILD_CONTEXT="$TEMP_SOURCE/source"
fi

if [[ -z "${NVFL_BASE_VERSION:-}" ]]; then
  NVFL_BASE_VERSION="$(python3 -c 'import re, nvflare; m = re.match(r"[0-9]+\.[0-9]+\.[0-9]+", nvflare.__version__); print(m.group(0) if m else "")')"
fi
[[ -n "$NVFL_BASE_VERSION" ]] || {
  echo "Unable to determine NVFL_BASE_VERSION; set it explicitly and retry." >&2
  exit 1
}

"$CONTAINER_TOOL" build --build-arg "NVFL_BASE_VERSION=$NVFL_BASE_VERSION" \
  -t "$PARENT_IMAGE" -f "$BUILD_CONTEXT/docker/Dockerfile.parent" "$BUILD_CONTEXT"
"$CONTAINER_TOOL" build --build-arg "NVFL_BASE_VERSION=$NVFL_BASE_VERSION" \
  -t "$WORKLOAD_IMAGE" -f "$BUILD_CONTEXT/docker/Dockerfile.job" "$BUILD_CONTEXT"
"$CONTAINER_TOOL" push "$PARENT_IMAGE"
"$CONTAINER_TOOL" push "$WORKLOAD_IMAGE"
