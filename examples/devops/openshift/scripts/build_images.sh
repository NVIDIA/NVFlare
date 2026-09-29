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
  git clone --quiet --filter=blob:none --no-checkout \
    "${NVFL_SOURCE_REPOSITORY:-https://github.com/NVIDIA/NVFlare.git}" "$TEMP_SOURCE/source"
  git -C "$TEMP_SOURCE/source" fetch --quiet origin "$REVISION"
  git -C "$TEMP_SOURCE/source" checkout --quiet --detach FETCH_HEAD
  BUILD_CONTEXT="$TEMP_SOURCE/source"
fi

"$CONTAINER_TOOL" build -t "$PARENT_IMAGE" -f "$BUILD_CONTEXT/docker/Dockerfile.parent" "$BUILD_CONTEXT"
"$CONTAINER_TOOL" build -t "$WORKLOAD_IMAGE" -f "$BUILD_CONTEXT/docker/Dockerfile.job" "$BUILD_CONTEXT"
"$CONTAINER_TOOL" push "$PARENT_IMAGE"
"$CONTAINER_TOOL" push "$WORKLOAD_IMAGE"
