#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: $0 platform-reference-values.json [--restart-rvps]" >&2; exit 2; }
[[ $# == 1 || "$2" == --restart-rvps ]] || { echo 'Unknown option' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALUES=$(realpath -e -- "$1")
PARSER="$SCRIPT_DIR/lib/platform-reference-values.py"
python3 "$PARSER" validate "$VALUES" >/dev/null
source "$SCRIPT_DIR/lib/common.sh"
require_sudo
need_file "$KBS_CLIENT"
need_file "$ADMIN_TOKEN"
need_file "$TRUSTEE_PUBLIC_CERT"
verify_references() {
    local key raw
    while IFS= read -r key; do
        raw=$(kbs_admin get-reference-value --id "$key")
        python3 "$PARSER" compare-reference "$VALUES" "$key" "$raw"
    done < <(python3 "$PARSER" reference-ids "$VALUES")
}
verify_references
if [[ ${2:-} == --restart-rvps ]]; then
    sudo docker compose -p "$TRUSTEE_PROJECT" -f "$TRUSTEE_COMPOSE" restart rvps
    # Retry only readiness; never retry or weaken a failed value comparison.
    for attempt in {1..30}; do
        if kbs_admin get-reference-value --id "$(python3 "$PARSER" reference-ids "$VALUES" | head -n1)" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    verify_references
    printf 'Reference persistence verified after RVPS restart.\n'
fi
printf 'All live RVPS references match the received complete approved set. No reference/policy settings changed.\n'
