#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || { echo 'Usage: build_coco_image.sh BUILD-REQUEST.json' >&2; exit 2; }
mapfile -d '' -t REQUEST < <(python3 - "$1" <<'PY'
import json
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[1]).read_text())
if set(request) != {"schema", "workload_env", "admin_dir", "result_file"} or request["schema"] != "nvflare-coco-build-request/v1":
    raise SystemExit("invalid CoCo build request")
values = [request[k] for k in ("workload_env", "admin_dir", "result_file")]
if any(not isinstance(v, str) or not Path(v).is_absolute() or "\0" in v for v in values):
    raise SystemExit("build request requires absolute paths")
if Path(values[2]).exists():
    raise SystemExit("result already exists; release is immutable")
sys.stdout.write("\0".join(values) + "\0")
PY
)
[[ ${#REQUEST[@]} -eq 3 ]] || { echo 'Invalid build request' >&2; exit 2; }
export OWNER_CONFIG="${REQUEST[0]}"
ADMIN_DIR="${REQUEST[1]}"
RESULT_FILE="${REQUEST[2]}"
# The platform file and scripts are trusted admin-node inputs, never CoCo inputs.
source "${ADMIN_DIR}/lib/release.sh"
for stage in 10-build-plaintext.sh 20-encrypt-sign-publish.sh 25-verify-published-image.sh \
    30-generate-pod-and-policies.sh 40-create-handoffs.sh; do
    [[ -x "${ADMIN_DIR}/${stage}" ]] || die "missing executable stage ${stage}"
done

"${ADMIN_DIR}/10-build-plaintext.sh" "${OWNER_CONFIG}"
[[ -t 0 ]] || die 'Review the plaintext image interactively before signing/publishing'
read -r -p "Review the image records above; type ${RELEASE_NAME} to sign, encrypt and publish: " APPROVAL
[[ "${APPROVAL}" == "${RELEASE_NAME}" ]] || die 'Plaintext image approval did not match'
"${ADMIN_DIR}/20-encrypt-sign-publish.sh" "${OWNER_CONFIG}" --approve-reviewed-plaintext
"${ADMIN_DIR}/25-verify-published-image.sh" "${OWNER_CONFIG}"
"${ADMIN_DIR}/30-generate-pod-and-policies.sh" "${OWNER_CONFIG}"
"${ADMIN_DIR}/40-create-handoffs.sh" "${OWNER_CONFIG}"

python3 - "${RESULT_FILE}" "${RELEASE_NAME}" "${HANDOFF_DIR}" <<'PY'
import json
from pathlib import Path
import sys

result, release, handoff = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
pod = handoff / "coco-it" / f"{release}-pod.yaml"
service = handoff / "trusted-service"
if not pod.is_file() or not (service / "image_key").is_file():
    raise SystemExit("handoff generation did not complete")
with result.open("x") as output:
    json.dump({"schema": "nvflare-coco-build-result/v1", "release_name": release,
               "pod_yaml": str(pod), "trusted_service": str(service)}, output, indent=2)
    output.write("\n")
PY
printf 'Deliver the trusted-service handoff separately to secure services; do not install it automatically.\n'
