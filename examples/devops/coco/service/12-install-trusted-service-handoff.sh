#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# -ge 1 && $# -le 2 ]] || {
    echo "Usage: $0 HANDOFF_DIRECTORY [EXPECTED_MANIFEST_SHA256]" >&2; exit 2;
}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"
[[ -n ${CURRENT_RESOURCE_POLICY_FILE:-} ]] || die 'Set CURRENT_RESOURCE_POLICY_FILE to the current resource-policy export.'
python3 "$SCRIPT_DIR/lib/install-workload-release.py" "$1" "$CURRENT_RESOURCE_POLICY_FILE" \
    "$STATE_DIR" "$KBS_CLIENT" "$KBS_URL" "$TRUSTEE_PUBLIC_CERT" "$ADMIN_TOKEN" "${2:-}"
