#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || { printf 'Usage: %s WORKLOAD.env\n' "$0" >&2; exit 2; }
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

need_file "${OUTPUT_DIR}/encrypted-image-reference.txt"
need_file "${OUTPUT_DIR}/cosign.pub"
need_file "${REGISTRY_AUTH_FILE}"
IMAGE_REF="$(tr -d '\r\n' < "${OUTPUT_DIR}/encrypted-image-reference.txt")"
KBS_KEY_URI="$(kbs_uri "${KBS_IMAGE_KEY_PATH}")"

DOCKER_CONFIG="${REGISTRY_SECRET_DIR}" cosign verify \
    --registry-cacert "${PUBLIC_DIR}/registry-ca.crt" \
    --key "${OUTPUT_DIR}/cosign.pub" --insecure-ignore-tlog \
    "${IMAGE_REF}" > "${OUTPUT_DIR}/owner-cosign-verification.json"
skopeo inspect --raw --authfile "${REGISTRY_AUTH_FILE}" "docker://${IMAGE_REF}" \
    > "${OUTPUT_DIR}/owner-encrypted-manifest.json"

python3 - "${OUTPUT_DIR}/owner-encrypted-manifest.json" "${KBS_KEY_URI}" <<'PY'
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
        raise SystemExit(f"layer {index} is not encrypted")
    value = layer.get("annotations", {}).get(name)
    if not value:
        raise SystemExit(f"layer {index} lacks key-provider metadata")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
    if expected not in decoded:
        raise SystemExit(f"layer {index} has the wrong KBS key ID")
print(f"owner verified {len(layers)} encrypted layer(s)")
PY

STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --cacert "${PUBLIC_DIR}/registry-ca.crt" --request POST \
    "https://$(registry_base)/v2/${REGISTRY_REPOSITORY}-unauthorized/blobs/uploads/")"
[[ "${STATUS}" == "401" || "${STATUS}" == "403" ]] \
    || die "anonymous registry write was not rejected (HTTP ${STATUS})"

printf 'Owner signature/encryption checks passed; anonymous write returned HTTP %s.\n' "${STATUS}"
printf 'Verify anonymous manifest/blob reads independently from a non-publisher origin such as coco.\n'
