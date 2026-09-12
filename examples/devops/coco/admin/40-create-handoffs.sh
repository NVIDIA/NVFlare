#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || { printf 'Usage: %s WORKLOAD.env\n' "$0" >&2; exit 2; }
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

for file in pod.yaml image-security-policy.json cosign.pub \
    release-authorization.json resource-policy-fragment.rego \
    expected-initdata-sha256.hex encrypted-image-reference.txt; do
    need_file "${OUTPUT_DIR}/${file}"
done
need_file "${IMAGE_KEY_PATH}"
need_file "${REGISTRY_PASSWORD_PATH}"
need_file "${COSIGN_PASSWORD_PATH}"
need_file "${COSIGN_KEY_PATH}"
need_file "${OUTPUT_DIR}/approved-workload-launch-profile.json"
python3 "${WORKLOAD_PROFILE_VALIDATOR}" \
    "${OUTPUT_DIR}/approved-workload-launch-profile.json" \
    "${WORKLOAD_LAUNCH_PROFILE_SHA256:-}" "$RUNTIME_CLASS" "$KATA_VERSION" \
    --pod "${OUTPUT_DIR}/pod.yaml"
[[ ! -e "${HANDOFF_DIR}" ]] || die "handoffs already exist; this release is immutable"

SERVICE_HANDOFF="${HANDOFF_DIR}/trusted-service"
COCO_HANDOFF="${HANDOFF_DIR}/coco-it"
mkdir -p "${SERVICE_HANDOFF}" "${COCO_HANDOFF}"
chmod 0700 "${HANDOFF_DIR}" "${SERVICE_HANDOFF}"
chmod 0755 "${COCO_HANDOFF}"

install -m 0600 "${IMAGE_KEY_PATH}" "${SERVICE_HANDOFF}/image_key"
install -m 0644 "${OUTPUT_DIR}/cosign.pub" "${SERVICE_HANDOFF}/cosign.pub"
install -m 0644 "${OUTPUT_DIR}/image-security-policy.json" \
    "${SERVICE_HANDOFF}/image-security-policy.json"
install -m 0644 "${OUTPUT_DIR}/release-authorization.json" \
    "${SERVICE_HANDOFF}/release-authorization.json"
install -m 0644 "${OUTPUT_DIR}/resource-policy-fragment.rego" \
    "${SERVICE_HANDOFF}/resource-policy-fragment.rego"

(
    cd "${SERVICE_HANDOFF}"
    sha256sum image_key cosign.pub image-security-policy.json \
        release-authorization.json resource-policy-fragment.rego > SHA256SUMS
)
chmod 0600 "${SERVICE_HANDOFF}/SHA256SUMS"

POD_NAME="${RELEASE_NAME}-pod.yaml"
install -m 0644 "${OUTPUT_DIR}/pod.yaml" "${COCO_HANDOFF}/${POD_NAME}"

python3 - "${COCO_HANDOFF}/${POD_NAME}" "${REGISTRY_PASSWORD_PATH}" \
    "${COSIGN_PASSWORD_PATH}" "${IMAGE_KEY_PATH}" "${COSIGN_KEY_PATH}" <<'PY'
from pathlib import Path
import sys

pod = Path(sys.argv[1]).read_bytes()
for secret_path in map(Path, sys.argv[2:]):
    secret = secret_path.read_bytes()
    if secret and secret in pod:
        raise SystemExit(f"Pod contains secret bytes from {secret_path}")
PY

COUNT="$(find "${COCO_HANDOFF}" -maxdepth 1 -type f | wc -l)"
[[ "${COUNT}" -eq 1 ]] || die "coco handoff must contain exactly one file"
POD_SHA256="$(sha256sum "${COCO_HANDOFF}/${POD_NAME}" | awk '{print $1}')"
printf '%s\n' "${POD_SHA256}" > "${HANDOFF_DIR}/expected-pod-sha256.txt"
chmod 0600 "${HANDOFF_DIR}/expected-pod-sha256.txt"

cat <<EOF
Handoffs created.

Trusted service administrator only (contains image_key):
  ${SERVICE_HANDOFF}

The only file to deliver manually to coco IT:
  ${COCO_HANDOFF}/${POD_NAME}

Authenticate this SHA-256 to IT over an independent channel:
  ${POD_SHA256}

Do not copy the trusted-service directory, registry credentials, signing key,
build context, plaintext image, or any other release file to coco.
EOF
