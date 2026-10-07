#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
CHANNEL="${NVFLARE_TDX_TCB_UPDATE_TYPE-early}"
case "${CHANNEL}" in
    early|standard) ;;
    *) printf 'NVFLARE_TDX_TCB_UPDATE_TYPE must be early or standard\n' >&2; exit 2 ;;
esac
IMAGE_ID='@VERIFIER_IMAGE_ID@'
[[ ${IMAGE_ID} =~ ^sha256:[0-9a-f]{64}$ ]] || { printf 'Run build.sh first\n' >&2; exit 1; }
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then DOCKER=(sudo -n docker); fi
OPTIONS=(--rm --read-only --cap-drop ALL --security-opt no-new-privileges
    --env "NVFLARE_TDX_TCB_UPDATE_TYPE=${CHANNEL}"
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=32m --user "$(id -u):$(id -g)")
if [[ $# == 1 && $1 == --version ]]; then
    exec "${DOCKER[@]}" run "${OPTIONS[@]}" --network none "${IMAGE_ID}" --version
fi
[[ $# == 3 ]] || {
    printf 'Usage: %s EVIDENCE.json CHALLENGE.bin INITDATA.toml\n' "$0" >&2; exit 2;
}
DESTINATIONS=(evidence.json challenge.bin initdata.toml)
for i in 0 1 2; do
    index=$((i + 1))
    source_path="$(realpath -- "${!index}")"
    [[ -f ${source_path} && -s ${source_path} && ${source_path} != *','* ]] || {
        printf 'Missing or unsupported input path: %s\n' "${source_path}" >&2; exit 1;
    }
    OPTIONS+=(--mount "type=bind,source=${source_path},target=/input/${DESTINATIONS[$i]},readonly")
done
# Network is required only for authenticated Intel PCS collateral retrieval.
# The verifier has no host device access, writes, capabilities or Docker socket.
exec "${DOCKER[@]}" run "${OPTIONS[@]}" "${IMAGE_ID}" \
    /input/evidence.json /input/challenge.bin /input/initdata.toml
