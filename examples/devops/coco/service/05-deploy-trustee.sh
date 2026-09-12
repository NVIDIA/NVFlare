#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_sudo
for command in docker git python3 ss; do need_cmd "${command}"; done
[[ "$(git -C "${TRUSTEE_ROOT}" rev-parse HEAD)" == "${TRUSTEE_COMMIT}" ]] || \
    die 'Trustee source is not the pinned post-v0.21 commit'
need_file "${TRUSTEE_ROOT}/built-image-ids.txt"
need_file "${KBS_CLIENT}"
for image in "${KBS_IMAGE}" "${AS_IMAGE}" "${RVPS_IMAGE}"; do
    sudo docker image inspect "${image}" >/dev/null
done

install -d -m 0700 "${KBS_STORAGE_DIR}" "${KBS_POLICY_DIR}" \
    "${AS_STORAGE_DIR}/attestation_service_policy" "${REFERENCE_DIR}"
install -m 0600 "${SCRIPT_DIR}/policies/default-deny-resource-policy.rego" \
    "${KBS_POLICY_DIR}/resource-policy.rego"

git -C "${TRUSTEE_ROOT}" show "${TRUSTEE_COMMIT}:docker-compose.yml" \
    > "${TRUSTEE_COMPOSE}"

# The pinned upstream snapshot still carries the pre-unified RVPS LocalFs
# shape (`type`/`file_path`). Current RVPS expects the unified storage backend
# shape; unknown legacy fields otherwise leave it on the default backend and
# the mounted reference-value directory remains empty.
need_file "${SCRIPT_DIR}/config/rvps-localfs.json"
install -m 0600 "${SCRIPT_DIR}/config/rvps-localfs.json" \
    "${TRUSTEE_ROOT}/kbs/config/docker-compose/rvps.json"

python3 - "${TRUSTEE_COMPOSE}" "${KBS_IMAGE}" "${AS_IMAGE}" \
    "${RVPS_IMAGE}" "${SETUP_IMAGE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
kbs, attestation_service, rvps, setup = sys.argv[2:]
text = path.read_text()
replacements = {
    "ghcr.io/confidential-containers/staged-images/kbs-grpc-as:latest": kbs,
    "ghcr.io/confidential-containers/staged-images/coco-as-grpc:latest": attestation_service,
    "ghcr.io/confidential-containers/staged-images/rvps:latest": rvps,
    "alpine/openssl": setup,
    '      - "8080:8080"': '      - "127.0.0.1:8080:8080"',
    '    - "50004:50004"': '    - "127.0.0.1:50004:50004"',
    '      - "50003:50003"': '      - "127.0.0.1:50003:50003"',
    "./kbs/data/attestation-service:/opt/confidential-containers/attestation-service:rw":
        "./kbs/data/attestation-service:/var/lib/attestation-service/storage:rw",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"unexpected Compose input: {old}")
    text = text.replace(old, new, 1)
repository = "./kbs/data/kbs-storage:/opt/confidential-containers/kbs/repository:rw"
policy = "./kbs/data/kbs-policy:/opt/confidential-containers/kbs/kbs:rw"
if text.count(repository) != 1:
    raise SystemExit("unexpected repository mount")
text = text.replace(repository, repository + "\n      - " + policy, 1)
path.write_text(text)
PY

# The upstream Compose token already carries aud=["KBS"]. Require that exact
# audience in the KBS trusted issuer so an admin JWT minted for another service
# cannot be replayed here.
python3 - "${TRUSTEE_ROOT}/kbs/config/docker-compose/kbs-config.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = '{ issuer = "TrusteeInDocker", public_key_uri = "/opt/confidential-containers/kbs/user-keys/public.pub" }'
new = '{ issuer = "TrusteeInDocker", audience = "KBS", public_key_uri = "/opt/confidential-containers/kbs/user-keys/public.pub" }'
if text.count(new) == 1 and old not in text:
    pass
elif text.count(old) == 1 and new not in text:
    path.write_text(text.replace(old, new, 1))
else:
    raise SystemExit("unexpected KBS admin trusted-issuer configuration")
PY

(
    cd "${TRUSTEE_ROOT}"
    sudo docker compose -p "${TRUSTEE_PROJECT}" up -d --pull never
)
for _ in $(seq 1 120); do
    if curl --fail --silent --show-error --output /dev/null \
        http://127.0.0.1:8080/healthz; then break; fi
    sleep 1
done
curl --fail --silent --show-error --output /dev/null http://127.0.0.1:8080/healthz

python3 - "${TRUSTEE_ROOT}/kbs/config/docker-compose/rvps.json" <<'PY'
from pathlib import Path
import json
import sys

config = json.loads(Path(sys.argv[1]).read_text())
expected = "/opt/confidential-containers/attestation-service/reference_values"
if config.get("storage", {}).get("storage_type") != "LocalFs":
    raise SystemExit("RVPS is not configured for LocalFs")
actual = (
    config.get("storage", {})
    .get("backends", {})
    .get("local_fs", {})
    .get("dir_path")
)
if actual != expected:
    raise SystemExit(f"unexpected RVPS LocalFs path: {actual!r}")
print("RVPS LocalFs configuration schema passed")
PY

need_file "${ADMIN_TOKEN}"
sudo chown root:root "${ADMIN_TOKEN}"
sudo chmod 0600 "${ADMIN_TOKEN}"
for port in 8080 50003 50004; do
    mapfile -t local_addresses < <(
        ss -ltnH "sport = :${port}" | awk '{print $4}'
    )
    (( ${#local_addresses[@]} > 0 )) || die "backend port ${port} is not listening"
    for address in "${local_addresses[@]}"; do
        [[ "${address}" == "127.0.0.1:${port}" ]] || \
            die "backend port ${port} has unexpected local bind ${address}"
    done
done
sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${TRUSTEE_COMPOSE}" ps
printf 'Pinned post-v0.21 Trustee is running on loopback backends.\n'
