#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl date docker grep mktemp python3 sha256sum; do
    need_cmd "${command}"
done
KBS_CONFIG="${TRUSTEE_ROOT}/kbs/config/docker-compose/kbs-config.toml"
ACTIVE_POLICY="${KBS_POLICY_DIR}/resource-policy.rego"
for file in "${KBS_CONFIG}" "${ACTIVE_POLICY}" "${ADMIN_TOKEN}" \
    "${TRUSTEE_COMPOSE}" "${TRUSTEE_PUBLIC_CERT}"; do
    need_file "${file}"
done

CHANGE_STATE="$(python3 - "${KBS_CONFIG}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = '{ issuer = "TrusteeInDocker", public_key_uri = "/opt/confidential-containers/kbs/user-keys/public.pub" }'
new = '{ issuer = "TrusteeInDocker", audience = "KBS", public_key_uri = "/opt/confidential-containers/kbs/user-keys/public.pub" }'
if text.count(new) == 1 and old not in text:
    print("already-hardened")
elif text.count(old) == 1 and new not in text:
    path.write_text(text.replace(old, new, 1))
    print("changed")
else:
    raise SystemExit("unexpected KBS admin trusted-issuer configuration")
PY
)"

if [[ "${CHANGE_STATE}" == "changed" ]]; then
    (
        cd "${TRUSTEE_ROOT}"
        sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${TRUSTEE_COMPOSE}" \
            up -d --no-deps --force-recreate kbs
    )
fi
wait_https "${KBS_URL}/healthz" "${TRUSTEE_PUBLIC_CERT}" 120

# Reinstall the exact same active bytes to exercise admin JWT validation while
# proving the workload authorization policy did not change.
POLICY_HASH_BEFORE="$(sha256sum "${ACTIVE_POLICY}" | awk '{print $1}')"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
kbs_admin set-resource-policy --policy-file "${ACTIVE_POLICY}" >/dev/null
POLICY_HASH_AFTER="$(sha256sum "${ACTIVE_POLICY}" | awk '{print $1}')"
[[ "${POLICY_HASH_BEFORE}" == "${POLICY_HASH_AFTER}" ]] \
    || die 'active resource policy changed during audience verification'

LOG_FILE="$(mktemp)"
trap 'rm -f -- "${LOG_FILE}"' EXIT
sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${TRUSTEE_COMPOSE}" \
    logs --no-color --since "${STARTED_AT}" kbs > "${LOG_FILE}" 2>&1
grep -Fq 'Endorsement of a token has been verified successfully.' "${LOG_FILE}" \
    || die 'admin JWT verification success was not observed'
if grep -Fq 'audience is not set' "${LOG_FILE}"; then
    die 'KBS still skips admin JWT audience verification'
fi

printf 'KBS admin JWT audience is pinned to KBS; resource policy hash is unchanged.\n'
