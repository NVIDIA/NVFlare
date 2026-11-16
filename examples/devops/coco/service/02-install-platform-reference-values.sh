#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 1 && $# -le 2 && ( ${2:-} == "" || ${2:-} == --verify-only ) ]] || {
    echo "Usage: $0 VALUES.json [--verify-only]" >&2; exit 2;
}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
source "$SCRIPT_DIR/lib/common.sh"
lock_platform_reference_update
WORK_DIR=$(mktemp -d "$STATE_DIR/platform-references.XXXXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
install -m 0600 "$VALUES" "$WORK_DIR/approved.json"
VALUES=$WORK_DIR/approved.json
PARSER=$SCRIPT_DIR/lib/platform-reference-values.py
python3 "$PARSER" validate "$VALUES" >/dev/null
if [[ ${2:-} != --verify-only ]]; then
    trap 'printf "Reference update failed; keep workloads stopped, correct the error and rerun the received file.\n" >&2' ERR
    python3 "$PARSER" install-references "$VALUES" "$KBS_URL" "$TRUSTEE_PUBLIC_CERT" "$ADMIN_TOKEN"
fi
verify_platform_references "$VALUES"
printf 'Live platform references match the received JSON; AS and workload resource policies were not changed.\n'
