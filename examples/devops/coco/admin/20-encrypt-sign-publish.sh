#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 2 && "$2" == "--approve-reviewed-plaintext" ]] || {
    printf 'Usage: %s WORKLOAD.env --approve-reviewed-plaintext\n' "$0" >&2
    exit 2
}
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

for command in cosign docker jq openssl python3 skopeo; do need_cmd "${command}"; done
need_file "${PRIVATE_DIR}/build-state.env"
need_file "${REGISTRY_USERNAME_PATH}"
need_file "${REGISTRY_PASSWORD_PATH}"

# shellcheck disable=SC1090
source "${PRIVATE_DIR}/build-state.env"
[[ "${CONFIG_SHA256}" == "$(sha256sum "${OWNER_CONFIG}" | awk '{print $1}')" ]] \
    || die "workload config changed after plaintext build; start a new release"
[[ "$(sudo docker image inspect --format '{{.Id}}' "${PLAIN_IMAGE}")" == "${PLAIN_IMAGE_ID}" ]] \
    || die "plaintext image changed after review"
[[ ! -e "${OUTPUT_DIR}/encrypted-image-reference.txt" ]] \
    || die "this release was already published"

[[ "$(tr -d '\r\n' < "${REGISTRY_USERNAME_PATH}")" == "${REGISTRY_USERNAME}" ]] \
    || die "registry username does not match the platform publisher identity"

mkdir -p "${PRIVATE_DIR}/plain-oci" "${PRIVATE_DIR}/encrypted-oci"
sudo skopeo copy --insecure-policy \
    "docker-daemon:${PLAIN_IMAGE}" "dir:${PRIVATE_DIR}/plain-oci"
sudo chown -R "$(id -u):$(id -g)" "${PRIVATE_DIR}"

if [[ ! -s "${IMAGE_KEY_PATH}" ]]; then
    openssl rand -out "${IMAGE_KEY_PATH}" 32
fi
[[ "$(wc -c < "${IMAGE_KEY_PATH}")" -eq 32 ]] || die "image key is not exactly 32 bytes"
chmod 0600 "${IMAGE_KEY_PATH}"

if [[ ! -s "${COSIGN_PASSWORD_PATH}" ]]; then
    openssl rand -base64 36 > "${COSIGN_PASSWORD_PATH}"
fi
chmod 0600 "${COSIGN_PASSWORD_PATH}"
if [[ -e "${COSIGN_KEY_PATH}" || -e "${COSIGN_PUB_PATH}" ]]; then
    need_file "${COSIGN_KEY_PATH}"
    need_file "${COSIGN_PUB_PATH}"
else
    COSIGN_PASSWORD="$(<"${COSIGN_PASSWORD_PATH}")" \
        cosign generate-key-pair --output-key-prefix "${SIGNING_DIR}/cosign"
fi
chmod 0600 "${COSIGN_KEY_PATH}" "${COSIGN_PASSWORD_PATH}"
chmod 0644 "${COSIGN_PUB_PATH}"

KBS_KEY_URI="$(kbs_uri "${KBS_IMAGE_KEY_PATH}")"
sudo docker pull "${KEYPROVIDER_IMAGE}"
sudo docker run --rm \
    --entrypoint /bin/sh \
    --volume "${IMAGE_KEY_PATH}:/key:ro" \
    --volume "${PRIVATE_DIR}:/work" \
    --env "KBS_KEY_URI=${KBS_KEY_URI}" \
    "${KEYPROVIDER_IMAGE}" \
    -ec '
        coco_keyprovider --socket 127.0.0.1:50000 &
        provider_pid=$!
        trap "kill ${provider_pid} >/dev/null 2>&1 || true" EXIT
        sleep 2
        skopeo copy --insecure-policy \
          --encryption-key "provider:attestation-agent:keypath=/key::keyid=${KBS_KEY_URI}::algorithm=A256GCM" \
          dir:/work/plain-oci dir:/work/encrypted-oci
    '
sudo chown -R "$(id -u):$(id -g)" "${PRIVATE_DIR}"

REGISTRY_PASSWORD="$(<"${REGISTRY_PASSWORD_PATH}")"
printf '%s' "${REGISTRY_PASSWORD}" | skopeo login \
    --authfile "${REGISTRY_AUTH_FILE}" \
    --username "${REGISTRY_USERNAME}" --password-stdin "$(registry_base)"
unset REGISTRY_PASSWORD
chmod 0600 "${REGISTRY_AUTH_FILE}"

IMAGE_TAG="${IMAGE_REPOSITORY}:${RELEASE_NAME}"
DIGEST_FILE="${PRIVATE_DIR}/encrypted.digest"
skopeo copy --insecure-policy --authfile "${REGISTRY_AUTH_FILE}" \
    --digestfile "${DIGEST_FILE}" \
    "dir:${PRIVATE_DIR}/encrypted-oci" "docker://${IMAGE_TAG}"

IMAGE_DIGEST="$(tr -d '\r\n' < "${DIGEST_FILE}")"
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "registry did not return a valid sha256 digest"
IMAGE_REF="${IMAGE_REPOSITORY}@${IMAGE_DIGEST}"

export DOCKER_CONFIG="${REGISTRY_SECRET_DIR}"
COSIGN_PASSWORD="$(<"${COSIGN_PASSWORD_PATH}")" \
    cosign sign --yes --tlog-upload=false --registry-cacert "${PUBLIC_DIR}/registry-ca.crt" \
        --key "${COSIGN_KEY_PATH}" "${IMAGE_REF}"
unset COSIGN_PASSWORD
cosign verify --registry-cacert "${PUBLIC_DIR}/registry-ca.crt" \
    --key "${COSIGN_PUB_PATH}" --insecure-ignore-tlog \
    "${IMAGE_REF}" > "${OUTPUT_DIR}/publisher-cosign-verification.json"

skopeo inspect --raw --authfile "${REGISTRY_AUTH_FILE}" \
    "docker://${IMAGE_REF}" > "${OUTPUT_DIR}/encrypted-manifest.json"

python3 - "${OUTPUT_DIR}/encrypted-manifest.json" "${KBS_KEY_URI}" <<'PY'
import base64
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2]
layers = manifest.get("layers", [])
if not layers:
    raise SystemExit("encrypted manifest has no layers")
name = "org.opencontainers.image.enc.keys.provider.attestation-agent"
for index, layer in enumerate(layers):
    if "+encrypted" not in layer.get("mediaType", ""):
        raise SystemExit(f"layer {index} is not marked encrypted")
    value = layer.get("annotations", {}).get(name)
    if not value:
        raise SystemExit(f"layer {index} lacks key-provider metadata")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
    if expected not in decoded:
        raise SystemExit(f"layer {index} key ID does not match {expected}")
print(f"verified {len(layers)} encrypted layer(s), all bound to {expected}")
PY

install -m 0644 "${COSIGN_PUB_PATH}" "${OUTPUT_DIR}/cosign.pub"
printf '%s\n' "${IMAGE_REF}" > "${OUTPUT_DIR}/encrypted-image-reference.txt"
printf '%s\n' "${IMAGE_TAG}" > "${OUTPUT_DIR}/encrypted-image-tag.txt"
printf '%s\n' "$(date -u +%FT%TZ)" > "${REVIEW_DIR}/plaintext-approved-at.txt"
sha256sum "${OUTPUT_DIR}/encrypted-manifest.json" "${OUTPUT_DIR}/cosign.pub" \
    > "${OUTPUT_DIR}/published-SHA256SUMS"

printf 'Published signed encrypted image: %s\n' "${IMAGE_REF}"
printf 'The plaintext image remains only on this trusted admin machine.\n'
