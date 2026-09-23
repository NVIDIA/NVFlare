#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

[[ $# -eq 2 ]] || {
    printf 'Usage: %s RELEASE-pod.yaml EXPECTED_SHA256\n' "$0" >&2
    exit 2
}
POD_FILE="$(realpath -- "$1")"
EXPECTED_SHA256="$2"
[[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die 'expected SHA-256 is not 64 lowercase hex characters'
need_file "${POD_FILE}"
for command in kubectl python3 sha256sum; do need_cmd "${command}"; done

ACTUAL_SHA256="$(sha256sum "${POD_FILE}" | awk '{print $1}')"
[[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256}" ]] || \
    die "Pod SHA-256 mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}"

POD_JSON="$(mktemp)"
trap 'rm -f "${POD_JSON}"' EXIT
kctl create --dry-run=client --validate=false -f "${POD_FILE}" -o json > "${POD_JSON}"
python3 - "${POD_JSON}" "${RUNTIME_CLASS}" "${REGISTRY_HOST}" \
    "${SCRIPT_DIR}/lib/workload-security-context.py" <<'PY'
import base64
import gzip
import json
from pathlib import Path
import re
import runpy
import sys
import tomllib

pod = json.loads(Path(sys.argv[1]).read_text())
runtime, registry = sys.argv[2:4]
helpers = runpy.run_path(sys.argv[4])
context = helpers["pod_context"](pod)
c = pod["spec"]["containers"][0]
image = c.get("image", "")
if not image.startswith(registry + "/") or not re.search(r"@sha256:[0-9a-f]{64}$", image):
    raise SystemExit("image must be an immutable digest in the approved registry")
if pod["spec"].get("runtimeClassName") != runtime:
    raise SystemExit("unexpected confidential runtime class")
if not isinstance(c.get("command"), list) or not c["command"]:
    raise SystemExit("an explicit command vector is required")
helpers["validate_workload_pod"](pod, context, c["command"])
print("Static handoff invariants passed")
print("Pod:", pod["metadata"]["name"])
print("Image:", image)
print("Command:", json.dumps(c["command"]))
PY

check_registry_trust "${REGISTRY_HOST}"

kctl apply --dry-run=server -f "${POD_FILE}" >/dev/null
printf 'Authenticated Pod SHA-256: %s\n' "${ACTUAL_SHA256}"
printf 'Do not edit this manifest. Do not use exec, attach, cp, or port-forward.\n'
confirm_action APPLY "Type APPLY within 120 seconds to launch the exact handoff: " ||
    die 'Approval missing, mismatched, timed out or input closed; launch cancelled'
kctl apply -f "${POD_FILE}"
POD_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["name"])' "${POD_JSON}")"
NAMESPACE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"].get("namespace", "default"))' "${POD_JSON}")"
kctl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=10m
kctl get pod "${POD_NAME}" -n "${NAMESPACE}" -o wide
