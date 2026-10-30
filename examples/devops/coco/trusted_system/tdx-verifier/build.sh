#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { printf 'Usage: %s OUTPUT_DIR\n' "$0" >&2; exit 2; }
ENGINE_NAME="${TDX_VERIFIER_ENGINE-docker}"
BUILD_NETWORK="${TDX_VERIFIER_BUILD_NETWORK:-default}"
case "${ENGINE_NAME}" in
    docker|podman) ;;
    *) printf 'TDX_VERIFIER_ENGINE must be docker or podman\n' >&2; exit 2 ;;
esac
case "${BUILD_NETWORK}" in
    default|host) ;;
    *) printf 'TDX_VERIFIER_BUILD_NETWORK must be default or host\n' >&2; exit 2 ;;
esac
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install -d -m 0700 -- "$1"
OUTPUT="$(realpath -- "$1")"
[[ ! -e "${OUTPUT}/tdx-evidence-verify" && ! -e "${OUTPUT}/tdx-evidence-verify.image-id" ]] || {
    printf 'Refusing to overwrite an approved verifier in %s\n' "${OUTPUT}" >&2; exit 1;
}
ENGINE=("${ENGINE_NAME}")
PULL=(--pull)
if [[ ${ENGINE_NAME} == docker ]]; then
    if ! docker info >/dev/null 2>&1; then ENGINE=(sudo -n docker); fi
    "${ENGINE[@]}" info >/dev/null
else
    [[ "$(podman info --format '{{.Host.Security.Rootless}}')" == true ]] || {
        printf 'Use rootless Podman as the capture user; sudo Podman is unsupported\n' >&2; exit 1;
    }
    PULL=(--pull=always)
    # Podman uses private for its default build network namespace.
    if [[ ${BUILD_NETWORK} == default ]]; then BUILD_NETWORK=private; fi
fi
[[ "$(uname -m)" == x86_64 ]] || { printf 'Intel DCAP verifier requires x86_64\n' >&2; exit 1; }
IMAGE="nvflare-tdx-evidence-verify:trustee-338610fbfed57b66"
"${ENGINE[@]}" build --network "${BUILD_NETWORK}" "${PULL[@]}" --platform linux/amd64 --tag "${IMAGE}" "${HERE}"
IMAGE_ID="$("${ENGINE[@]}" image inspect --format '{{.Id}}' "${IMAGE}")"
if [[ ${ENGINE_NAME} == podman && ${IMAGE_ID} =~ ^[0-9a-f]{64}$ ]]; then IMAGE_ID="sha256:${IMAGE_ID}"; fi
[[ ${IMAGE_ID} =~ ^sha256:[0-9a-f]{64}$ ]] || {
    printf 'Container engine returned an invalid verifier image ID\n' >&2; exit 1;
}
"${ENGINE[@]}" run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges "${IMAGE_ID}" --version
# Embed the immutable image ID in the installed wrapper: its SHA256 then
# covers which cryptographic verifier is executed, not just launcher code.
python3 - "${HERE}/run-verifier.sh" "${OUTPUT}/tdx-evidence-verify" "${IMAGE_ID}" "${ENGINE_NAME}" "$(id -u)" <<'PY'
import os
import sys
from pathlib import Path

source, destination, image_id, engine, uid = sys.argv[1:]
text = (
    Path(source)
    .read_text()
    .replace("@VERIFIER_IMAGE_ID@", image_id)
    .replace("@VERIFIER_ENGINE@", engine)
    .replace("@VERIFIER_BUILD_UID@", uid)
)
with open(destination, "x", encoding="utf-8") as stream:
    stream.write(text)
os.chmod(destination, 0o700)
PY
printf '%s\n' "${IMAGE_ID}" > "${OUTPUT}/tdx-evidence-verify.image-id"
printf 'Installed pinned TDX verifier: %s\n' "${OUTPUT}/tdx-evidence-verify"
