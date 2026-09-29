#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { printf 'Usage: %s OUTPUT_DIR\n' "$0" >&2; exit 2; }
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install -d -m 0700 -- "$1"
OUTPUT="$(realpath -- "$1")"
[[ ! -e "${OUTPUT}/tdx-evidence-verify" && ! -e "${OUTPUT}/tdx-evidence-verify.image-id" ]] || {
    printf 'Refusing to overwrite an approved verifier in %s\n' "${OUTPUT}" >&2; exit 1;
}
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then DOCKER=(sudo -n docker); fi
"${DOCKER[@]}" info >/dev/null
[[ "$(uname -m)" == x86_64 ]] || { printf 'Intel DCAP verifier requires x86_64\n' >&2; exit 1; }
IMAGE="nvflare-tdx-evidence-verify:trustee-338610fbfed57b66"
BUILD_NETWORK="${TDX_VERIFIER_BUILD_NETWORK:-default}"
case "${BUILD_NETWORK}" in
    default|host) ;;
    *) printf 'TDX_VERIFIER_BUILD_NETWORK must be default or host\n' >&2; exit 2 ;;
esac
"${DOCKER[@]}" build --network "${BUILD_NETWORK}" --pull --platform linux/amd64 --tag "${IMAGE}" "${HERE}"
IMAGE_ID="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "${IMAGE}")"
[[ ${IMAGE_ID} =~ ^sha256:[0-9a-f]{64}$ ]]
"${DOCKER[@]}" run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges "${IMAGE_ID}" --version
# Embed the immutable image ID in the installed wrapper: its SHA256 then
# covers which cryptographic verifier is executed, not just launcher code.
python3 - "${HERE}/run-verifier.sh" "${OUTPUT}/tdx-evidence-verify" "${IMAGE_ID}" <<'PY'
import os
import sys
from pathlib import Path

source, destination, image_id = sys.argv[1:]
text = Path(source).read_text().replace("@VERIFIER_IMAGE_ID@", image_id)
with open(destination, "x", encoding="utf-8") as stream:
    stream.write(text)
os.chmod(destination, 0o700)
PY
printf '%s\n' "${IMAGE_ID}" > "${OUTPUT}/tdx-evidence-verify.image-id"
printf 'Installed pinned TDX verifier: %s\n' "${OUTPUT}/tdx-evidence-verify"
