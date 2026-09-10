#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || { printf 'Usage: %s WORKLOAD.env\n' "$0" >&2; exit 2; }
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

need_cmd docker
need_cmd python3

[[ ! -e "${RELEASE_DIR}" ]] \
    || die "release already exists and is immutable: ${RELEASE_DIR}; choose a new RELEASE_NAME"

mkdir -p "${PRIVATE_DIR}" "${REVIEW_DIR}" "${OUTPUT_DIR}"
chmod 0700 "${RELEASE_DIR}" "${PRIVATE_DIR}"
install -m 0600 "${OWNER_CONFIG}" "${PRIVATE_DIR}/workload.env"

BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PLAIN_IMAGE="localhost/coco-owner/${RELEASE_NAME}:plaintext-${BUILD_ID}"

sudo docker build --pull --no-cache --file "${DOCKERFILE}" \
    --tag "${PLAIN_IMAGE}" "${BUILD_CONTEXT}"

sudo docker image inspect "${PLAIN_IMAGE}" > "${REVIEW_DIR}/plaintext-image-inspect.json"
sudo docker history --no-trunc "${PLAIN_IMAGE}" > "${REVIEW_DIR}/plaintext-image-history.txt"
sudo chown -R "$(id -u):$(id -g)" "${RELEASE_DIR}"

PLAIN_IMAGE_ID="$(sudo docker image inspect --format '{{.Id}}' "${PLAIN_IMAGE}")"
[[ "${PLAIN_IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "unexpected plaintext image ID"

{
    printf 'RELEASE_NAME=%q\n' "${RELEASE_NAME}"
    printf 'BUILD_ID=%q\n' "${BUILD_ID}"
    printf 'PLAIN_IMAGE=%q\n' "${PLAIN_IMAGE}"
    printf 'PLAIN_IMAGE_ID=%q\n' "${PLAIN_IMAGE_ID}"
    printf 'CONFIG_SHA256=%q\n' "${CONFIG_SHA256}"
} > "${PRIVATE_DIR}/build-state.env"
chmod 0600 "${PRIVATE_DIR}/build-state.env"

sha256sum "${DOCKERFILE}" "${PRIVATE_DIR}/workload.env" \
    "${REVIEW_DIR}/plaintext-image-inspect.json" \
    "${REVIEW_DIR}/plaintext-image-history.txt" \
    > "${REVIEW_DIR}/SHA256SUMS"

cat <<EOF
Built plaintext image ${PLAIN_IMAGE_ID}.

STOP AND REVIEW:
  ${REVIEW_DIR}/plaintext-image-inspect.json
  ${REVIEW_DIR}/plaintext-image-history.txt

Check for secrets, unwanted ENTRYPOINT/CMD/ENV/history, sshd, shells, package
managers, and a UID/GID mismatch. Run functional and vulnerability tests here.
Only after approval run:
  ${SCRIPT_DIR}/20-encrypt-sign-publish.sh ${OWNER_CONFIG} --approve-reviewed-plaintext
EOF
