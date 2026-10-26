#!/usr/bin/env bash
# Validate the received manifest, then apply that same private snapshot.
set -Eeuo pipefail
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# -ge 1 && $# -le 2 ]] || {
    printf 'Usage: %s POD.yaml [EXPECTED_SHA256]\n' "$0" >&2
    exit 2
}
EXPECTED_SHA256="${2:-}"
if [[ -n "${EXPECTED_SHA256}" ]]; then
    [[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die 'expected SHA-256 is not 64 lowercase hex characters'
fi
for executable in kubectl python3; do
    command -v "${executable}" >/dev/null || die "required command is missing: ${executable}"
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "${WORK_DIR}"' EXIT
POD_SNAPSHOT="${WORK_DIR}/pod.yaml"
POD_JSON="${WORK_DIR}/pod.json"
python3 - "$1" "${POD_SNAPSHOT}" "${POD_JSON}" "${EXPECTED_SHA256}" \
    "${SCRIPT_DIR}/lib/workload-security-context.py" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import stat
import sys

source, snapshot, metadata = map(Path, sys.argv[1:4])
with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        raise SystemExit("Pod input must be a regular file")
    payload = stream.read()
if sys.argv[4] and hashlib.sha256(payload).hexdigest() != sys.argv[4]:
    raise SystemExit("Pod SHA-256 mismatch")
snapshot.write_bytes(payload)
helpers = runpy.run_path(sys.argv[5])
pod = helpers["read_pod"](snapshot)
context = helpers["pod_context"](pod)
container = pod["spec"]["containers"][0]
image = container.get("image", "")
if not isinstance(image, str) or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
    raise SystemExit("image must use an immutable SHA256 digest")
command = container.get("command")
if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
    raise SystemExit("an explicit command vector is required")
helpers["validate_workload_pod"](pod, context, command, runtime_class=pod["spec"].get("runtimeClassName"))
metadata.write_text(json.dumps(pod))
print("Received Pod passed image, runtime, security-context and InitData policy checks")
PY

kubectl apply --dry-run=server -f "${POD_SNAPSHOT}" >/dev/null
kubectl apply -f "${POD_SNAPSHOT}"
POD_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["name"])' "${POD_JSON}")"
NAMESPACE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"].get("namespace", "default"))' "${POD_JSON}")"
kubectl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=10m
kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" -o wide
printf 'Pod Ready; verify Trustee key release and NVFlare registration and peer proofs from the trusted side.\n'
