#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=platform.sh
source "${LIB_DIR}/platform.sh"

readonly WORK_ROOT
readonly KATA_VERSION KEYPROVIDER_IMAGE GENPOLICY RULES SETTINGS
readonly RUNTIME_CLASS KUBERNETES_SERVICE_HOST KUBERNETES_SERVICE_PORT
WORKLOAD_LAUNCH_PROFILE_SHA256="${WORKLOAD_LAUNCH_PROFILE_SHA256:-}"
readonly WORKLOAD_LAUNCH_PROFILE_SHA256
readonly WORKLOAD_LAUNCH_PROFILE
readonly WORKLOAD_PROFILE_VALIDATOR="${LIB_DIR}/workload-launch-profile.py"

[[ -n "${OWNER_CONFIG:-}" ]] || die "OWNER_CONFIG was not set by the caller"
OWNER_CONFIG="$(readlink -f -- "${OWNER_CONFIG}")"
need_file "${OWNER_CONFIG}"
# shellcheck disable=SC1090
source "${OWNER_CONFIG}"

# The workload request selects a target but cannot override platform authority.
# Reject a mismatch before building/signing/publishing anything.
case "$RUNTIME_CLASS" in
    kata-qemu-nvidia-gpu-snp|kata-qemu-nvidia-gpu-tdx) EXPECTED_GPU_COUNT=1 ;;
    kata-qemu-snp|kata-qemu-tdx) EXPECTED_GPU_COUNT=0 ;;
    *) die "unsupported approved RUNTIME_CLASS: $RUNTIME_CLASS" ;;
esac
[[ "${COCO_RUNTIME_CLASS:-$RUNTIME_CLASS}" == "$RUNTIME_CLASS" ]] \
    || die 'requested COCO_RUNTIME_CLASS differs from the approved platform runtime'
[[ "${COCO_GPU_COUNT:-$EXPECTED_GPU_COUNT}" == "$EXPECTED_GPU_COUNT" ]] \
    || die 'requested COCO_GPU_COUNT differs from the approved platform runtime'
readonly EXPECTED_GPU_COUNT

# Explicitly bound into genpolicy/InitData. Existing workloads stay read-only;
# provisioned NVFlare clients require guest-local writable logs and job state.
APP_READ_ONLY_ROOT_FILESYSTEM="${APP_READ_ONLY_ROOT_FILESYSTEM:-true}"
[[ "${APP_READ_ONLY_ROOT_FILESYSTEM}" == true || "${APP_READ_ONLY_ROOT_FILESYSTEM}" == false ]] \
    || die "APP_READ_ONLY_ROOT_FILESYSTEM must be true or false"
KBS_CA_FILE="${KBS_CA_FILE:-${PUBLIC_DIR}/trustee.crt}"

for required in RELEASE_NAME BUILD_CONTEXT DOCKERFILE REGISTRY_REPOSITORY \
    REGISTRY_ENDPOINT REGISTRY_CA_FILE REGISTRY_USERNAME_FILE REGISTRY_PASSWORD_FILE \
    KBS_URL KBS_CA_FILE APP_COMMAND_JSON APP_UID APP_GID; do
    [[ -n "${!required:-}" ]] || die "missing ${required} in ${OWNER_CONFIG}"
done

validate_registry_host "${REGISTRY_ENDPOINT}"
REGISTRY_USERNAME_PATH="$(readlink -f -- "${REGISTRY_USERNAME_FILE}")"
REGISTRY_PASSWORD_PATH="$(readlink -f -- "${REGISTRY_PASSWORD_FILE}")"
REGISTRY_CA_FILE="$(readlink -f -- "${REGISTRY_CA_FILE}")"
KBS_CA_FILE="$(readlink -f -- "${KBS_CA_FILE}")"
need_file "${REGISTRY_USERNAME_PATH}"
need_file "${REGISTRY_PASSWORD_PATH}"
need_file "${REGISTRY_CA_FILE}"
need_file "${KBS_CA_FILE}"
REGISTRY_USERNAME="$(tr -d '\r\n' < "${REGISTRY_USERNAME_PATH}")"
[[ -n "${REGISTRY_USERNAME}" ]] || die "registry username is empty"
readonly REGISTRY_ENDPOINT REGISTRY_USERNAME KBS_URL KBS_CA_FILE

registry_base() {
    printf '%s' "${REGISTRY_ENDPOINT}"
}

[[ "${RELEASE_NAME}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] \
    || die "RELEASE_NAME must be a lowercase DNS label"
(( ${#RELEASE_NAME} <= 63 )) || die "RELEASE_NAME must be at most 63 characters"
[[ "${REGISTRY_REPOSITORY}" =~ ^[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*$ ]] \
    || die "REGISTRY_REPOSITORY is not a valid lowercase repository"
[[ "${APP_UID}" =~ ^[0-9]+$ && "${APP_UID}" != 0 ]] || die "APP_UID must be a nonzero integer"
[[ "${APP_GID}" =~ ^[0-9]+$ && "${APP_GID}" != 0 ]] || die "APP_GID must be a nonzero integer"

BUILD_CONTEXT="$(readlink -f -- "${BUILD_CONTEXT}")"
DOCKERFILE="$(readlink -f -- "${DOCKERFILE}")"
[[ -d "${BUILD_CONTEXT}" ]] || die "BUILD_CONTEXT is not a directory: ${BUILD_CONTEXT}"
need_file "${DOCKERFILE}"

APP_COMMAND_JSON="$({
    python3 - "${APP_COMMAND_JSON}" <<'PY'
import json
import sys

try:
    value = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"APP_COMMAND_JSON is invalid JSON: {exc}")
if not isinstance(value, list) or not value:
    raise SystemExit("APP_COMMAND_JSON must be a non-empty JSON array")
if any(not isinstance(item, str) or not item for item in value):
    raise SystemExit("every APP_COMMAND_JSON item must be a non-empty string")
if not value[0].startswith("/"):
    raise SystemExit("the first APP_COMMAND_JSON item must be an absolute path")
print(json.dumps(value, separators=(",", ":")))
PY
} )"

RELEASE_DIR="${RELEASES_DIR}/${RELEASE_NAME}"
PRIVATE_DIR="${RELEASE_DIR}/private"
REVIEW_DIR="${RELEASE_DIR}/review"
OUTPUT_DIR="${RELEASE_DIR}/output"
HANDOFF_DIR="${RELEASE_DIR}/handoff"

IMAGE_KEY_PATH="${PRIVATE_DIR}/image_key"
COSIGN_KEY_PATH="${SIGNING_DIR}/cosign.key"
COSIGN_PUB_PATH="${SIGNING_DIR}/cosign.pub"
COSIGN_PASSWORD_PATH="${SIGNING_DIR}/cosign.password"
REGISTRY_SECRET_DIR="${PRIVATE_DIR}/registry-auth"
REGISTRY_AUTH_FILE="${REGISTRY_SECRET_DIR}/config.json"
REGISTRY_CERT_DIR="${PRIVATE_DIR}/registry-certs"

KBS_IMAGE_KEY_PATH="default/image-key/${RELEASE_NAME}"
KBS_SIGNING_KEY_PATH="default/sig-public-key/${RELEASE_NAME}"
KBS_IMAGE_POLICY_PATH="default/security-policy/${RELEASE_NAME}"
IMAGE_REPOSITORY="$(registry_base)/${REGISTRY_REPOSITORY}"

CONFIG_SHA256="$(sha256sum "${OWNER_CONFIG}" | awk '{print $1}')"

need_file "${REGISTRY_CA_FILE}"

# Called only by publication/verification, after stage 10 creates this release.
# Skopeo reads these roots for this registry operation; system trust is unchanged.
prepare_registry_certificates() {
    mkdir -p "${REGISTRY_CERT_DIR}"
    chmod 0700 "${REGISTRY_CERT_DIR}"
    install -m 0600 "${REGISTRY_CA_FILE}" "${REGISTRY_CERT_DIR}/ca.crt"
}
