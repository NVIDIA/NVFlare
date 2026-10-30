#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in curl docker openssl python3; do
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

# The read-only endpoint must accept the control and reject equally valid
# signatures with a wrong/missing audience. No resource policy is rewritten.
as_root python3 "${SCRIPT_DIR}/lib/kbs-admin-audience.py" "${KBS_URL}" \
    "${TRUSTEE_PUBLIC_CERT}" "${TRUSTEE_ROOT}/kbs/config/docker-compose/private.key"
