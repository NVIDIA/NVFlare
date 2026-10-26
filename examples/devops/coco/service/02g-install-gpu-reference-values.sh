#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 1 && $# -le 2 && ( ${2:-} == "" || ${2:-} == --verify-only || ${2:-} == --approve-gpu-reference-values ) ]] || {
    echo "Usage: $0 GPU-VALUES.json [--verify-only]" >&2; exit 2;
}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
source "$SCRIPT_DIR/lib/common.sh"
lock_platform_reference_update
WORK_DIR=$(mktemp -d "$STATE_DIR/gpu-references.XXXXXXXX")
STAGED=0
cleanup() {
    local status=$?
    if [[ $STAGED == 1 ]]; then
        if post_stage clear && verify_stage clear; then
            printf 'GPU driver approval was cleared after failure; VBIOS may have changed.\n' >&2
        else
            printf 'GPU approval state is UNKNOWN: compensating driver clear/readback failed. Stop GPU releases and inspect RVPS.\n' >&2
        fi
    fi
    rm -rf -- "$WORK_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
install -m 0600 "$VALUES" "$WORK_DIR/approved.json"
VALUES=$WORK_DIR/approved.json
PARSER=$SCRIPT_DIR/lib/gpu-reference-values.py
python3 "$PARSER" validate "$VALUES" >/dev/null
verify_stage() {
    local reference_id raw
    reference_id=$(python3 "$PARSER" reference-id "$VALUES" "$1")
    raw=$(kbs_admin get-reference-value --id "$reference_id")
    python3 "$PARSER" compare-stage "$VALUES" "$1" "$raw"
}
post_stage() {
    python3 "$PARSER" post-stage "$VALUES" "$1" "$KBS_URL" "$TRUSTEE_PUBLIC_CERT" "$ADMIN_TOKEN"
}
if [[ ${2:-} != --verify-only ]]; then
    STAGED=1
    post_stage clear
    verify_stage clear
    post_stage vbios
    verify_stage vbios
    post_stage drivers
    verify_stage drivers
fi
verify_gpu_references "$VALUES"
STAGED=0
printf 'Live GPU references match the received JSON; AS and workload resource policies were not changed.\n'
