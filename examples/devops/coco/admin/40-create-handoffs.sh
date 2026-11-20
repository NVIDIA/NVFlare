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
need_file "${OUTPUT_DIR}/policy-SHA256SUMS"
python3 - "${OUTPUT_DIR}" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import stat
import sys

root = Path(sys.argv[1])
names = {
    "approved-workload-launch-profile.json", "pod.yaml", "final-initdata.toml",
    "generated-policy.rego", "image-security-policy.json", "release-authorization.json",
    "resource-policy-fragment.rego", "kata-policy-security-fix.txt",
}
def read_regular(path, limit=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SystemExit("Release output must be a regular file")
        raw = stream.read(limit + 1) if limit is not None else stream.read()
        if limit is not None and len(raw) > limit:
            raise SystemExit("Release checksum manifest is too large")
        return raw

entries = {}
for line in read_regular(root / "policy-SHA256SUMS", 8192).decode().splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", line)
    if match is None or match[2] not in names or match[2] in entries:
        raise SystemExit("Invalid release checksum manifest")
    entries[match[2]] = match[1]
if set(entries) != names:
    raise SystemExit("Release checksum manifest must cover all generated policy outputs")
for name, digest in entries.items():
    if hashlib.sha256(read_regular(root / name)).hexdigest() != digest:
        raise SystemExit("Generated release output changed: " + name)
PY
python3 "${WORKLOAD_PROFILE_VALIDATOR}" \
    "${OUTPUT_DIR}/approved-workload-launch-profile.json" \
    "${WORKLOAD_LAUNCH_PROFILE_SHA256:-}" "$RUNTIME_CLASS" "$KATA_VERSION" \
    --pod "${OUTPUT_DIR}/pod.yaml" --require-policy
python3 - "${SCRIPT_DIR}/lib/workload-release.py" "${OUTPUT_DIR}" \
    "$RUNTIME_CLASS" "$RELEASE_NAME" "$APP_COMMAND_JSON" <<'PY'
import base64
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
import yaml

contract = runpy.run_path(sys.argv[1])
output = Path(sys.argv[2])
auth = contract["load_authorization"](output / "release-authorization.json")
pod = yaml.safe_load((output / "pod.yaml").read_text())
raw = gzip.decompress(base64.b64decode(
    pod["metadata"]["annotations"]["io.katacontainers.config.hypervisor.cc_init_data"], validate=True
))
if (auth["cpu_tee"], auth["gpu"]) != contract["target_for_runtime"](sys.argv[3]):
    raise SystemExit("release authorization differs from approved runtime")
if auth["release_name"] != sys.argv[4] or auth["process_args"] != json.loads(sys.argv[5]):
    raise SystemExit("release authorization differs from approved workload")
if auth["init_data_sha256"] != hashlib.sha256(raw).hexdigest():
    raise SystemExit("release authorization InitData digest differs from Pod")
if auth["encrypted_image"] != pod["spec"]["containers"][0]["image"]:
    raise SystemExit("release authorization image differs from Pod")
PY
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

cat <<EOF
Handoffs created.

Trusted service administrator only (contains image_key):
  ${SERVICE_HANDOFF}

The only file to deliver manually to coco IT:
  ${COCO_HANDOFF}/${POD_NAME}

Do not copy the trusted-service directory, registry credentials, signing key,
build context, plaintext image, or any other release file to coco.
EOF
