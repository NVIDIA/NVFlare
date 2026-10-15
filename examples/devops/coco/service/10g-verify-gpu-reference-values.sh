#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: $0 GPU-VALUES.json [--restart-rvps]" >&2; exit 2; }
[[ $# == 1 || "$2" == --restart-rvps ]] || { echo 'Unknown option' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
PARSER="$SCRIPT_DIR/lib/gpu-reference-values.py"
python3 "$PARSER" validate "$VALUES"
source "$SCRIPT_DIR/lib/common.sh"
if [[ ${2:-} == --restart-rvps ]]; then
    lock_platform_reference_update
fi
require_sudo
for file in "$KBS_CLIENT" "$ADMIN_TOKEN" "$TRUSTEE_PUBLIC_CERT"; do need_file "$file"; done

verify_references() {
    local stage reference_id raw
    for stage in vbios drivers; do
        reference_id=$(python3 "$PARSER" reference-id "$VALUES" "$stage")
        raw=$(kbs_admin get-reference-value --id "$reference_id")
        python3 "$PARSER" compare-stage "$VALUES" "$stage" "$raw"
    done
}
verify_references
if [[ ${2:-} == --restart-rvps ]]; then
    sudo docker compose -p "$TRUSTEE_PROJECT" -f "$TRUSTEE_COMPOSE" restart rvps
    # Retry readiness only; a mismatched value must fail immediately.
    for attempt in {1..30}; do
        if kbs_admin get-reference-value --id allowed_driver_versions >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    verify_references
    printf 'GPU reference persistence verified after RVPS restart.\n'
fi
printf 'Both live GPU version reference lists match the independently approved file.\n'
