#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

need_file "${SCRIPT_DIR}/config.env.example"
install -d -m 0700 "$(dirname -- "${COCO_CONFIG}")" "${COCO_STATE_DIR}"
if [[ ! -e "${COCO_CONFIG}" ]]; then
    install -m 0600 "${SCRIPT_DIR}/config.env.example" "${COCO_CONFIG}"
    printf 'Created %s. Review NODE_IP, TEE_PLATFORM, runtime, and all pins, then rerun.\n' \
        "${COCO_CONFIG}" >&2
    exit 2
fi
run_upstream_stage 00-verify-host.sh
