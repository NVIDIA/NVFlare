#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common-base.sh"

set -Eeuo pipefail
umask 077

KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../platform.env
[[ -s "${KIT_ROOT}/platform.env" ]] || { echo 'Copy platform.env.example to platform.env, chmod 600, and review it first.' >&2; exit 2; }
source "${KIT_ROOT}/platform.env"
source "${KIT_ROOT}/lib/validate-config.sh"
validate_target_host
validate_service_host "$SERVICE_FQDN"
[[ $REGISTRY_PORT == 5000 && $REGISTRY_HOST == "${SERVICE_FQDN}:5000" ]] || config_error 'Use registry TLS port 5000 on the configured secure-services DNS name.'
COCO_WORKFLOW="${KIT_ROOT}/bootstrap"

run_upstream_stage() {
    local stage="$1"
    need_file "${COCO_WORKFLOW}/${stage}"
    COCO_CONFIG="${COCO_CONFIG}" COCO_STATE_DIR="${COCO_STATE_DIR}" \
        "${COCO_WORKFLOW}/${stage}"
}
