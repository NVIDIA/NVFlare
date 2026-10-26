#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$SCRIPT_ROOT/lib/common-base.sh"
[[ -s $SCRIPT_ROOT/platform.env ]] || die 'Copy platform.env.example to platform.env and configure existing Trustee access.'
source "$SCRIPT_ROOT/platform.env"
for variable in KBS_URL TRUSTEE_PUBLIC_CERT ADMIN_TOKEN KBS_CLIENT; do
    [[ -n ${!variable:-} ]] || die "Set $variable for the existing Trustee."
done
STATE_DIR=${STATE_DIR:-${HOME}/.local/state/nvflare-coco-trustee}
need_file "$TRUSTEE_PUBLIC_CERT"
need_file "$ADMIN_TOKEN"
need_cmd "$KBS_CLIENT"
# Credentials are opaque to this kit. The existing Trustee validates their
# signatures, expiry, audience and administration permissions.
python3 - "$KBS_URL" <<'URL'
import sys
from urllib.parse import urlsplit
url = urlsplit(sys.argv[1])
if (url.scheme != "https" or not url.hostname or url.username or url.password
        or url.path not in ("", "/") or url.query or url.fragment):
    raise SystemExit("KBS_URL must be an HTTPS origin without embedded credentials")
if url.port is not None and not 1 <= url.port <= 65535:
    raise SystemExit("Invalid KBS port")
URL
KBS_URL=${KBS_URL%/}
install -d -m 0700 "$STATE_DIR"

kbs_admin() {
    local output
    if output=$("$KBS_CLIENT" --url "$KBS_URL" --cert-file "$TRUSTEE_PUBLIC_CERT" \
        config --admin-token-file "$ADMIN_TOKEN" "$@" 2>/dev/null); then
        printf '%s\n' "$output"
    else
        die 'Trustee administration request failed; check the endpoint and credential permissions.'
    fi
}

lock_platform_reference_update() {
    need_cmd flock
    exec {PLATFORM_REFERENCE_LOCK_FD}>"$STATE_DIR/reference-update.lock"
    flock -n "$PLATFORM_REFERENCE_LOCK_FD" || die 'Another reference update is in progress.'
}

verify_platform_references() {
    local values=$1 key raw
    while IFS= read -r key; do
        raw=$(kbs_admin get-reference-value --id "$key")
        python3 "$SCRIPT_ROOT/lib/platform-reference-values.py" compare-reference "$values" "$key" "$raw"
    done < <(python3 "$SCRIPT_ROOT/lib/platform-reference-values.py" reference-ids "$values")
}

verify_gpu_references() {
    local values=$1 stage reference_id raw
    for stage in vbios drivers; do
        reference_id=$(python3 "$SCRIPT_ROOT/lib/gpu-reference-values.py" reference-id "$values" "$stage")
        raw=$(kbs_admin get-reference-value --id "$reference_id")
        python3 "$SCRIPT_ROOT/lib/gpu-reference-values.py" compare-stage "$values" "$stage" "$raw"
    done
}
