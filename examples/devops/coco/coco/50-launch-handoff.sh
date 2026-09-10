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
python3 - "${POD_JSON}" "${RUNTIME_CLASS}" "${REGISTRY_HOST}" <<'PY'
import base64
import gzip
import json
from pathlib import Path
import re
import sys

pod = json.loads(Path(sys.argv[1]).read_text())
runtime, registry = sys.argv[2:]
if pod.get("kind") != "Pod" or pod.get("apiVersion") != "v1":
    raise SystemExit("handoff must contain exactly one v1 Pod")
spec = pod.get("spec", {})
containers = spec.get("containers", [])
if len(containers) != 1:
    raise SystemExit("handoff must have exactly one container")
c = containers[0]
image = c.get("image", "")
if not image.startswith(registry + "/") or not re.search(r"@sha256:[0-9a-f]{64}$", image):
    raise SystemExit("image must be an immutable digest in the approved registry")
if spec.get("runtimeClassName") != runtime:
    raise SystemExit("unexpected confidential runtime class")
if spec.get("automountServiceAccountToken") is not False:
    raise SystemExit("service-account token must be disabled")
if spec.get("enableServiceLinks") is not False:
    raise SystemExit("Kubernetes service-link environment injection must be disabled")
for key in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace"):
    if key == "shareProcessNamespace" and key not in spec:
        continue
    if spec.get(key) is not False:
        raise SystemExit(f"{key} must be false")
if "volumes" in spec:
    raise SystemExit("volumes are not allowed in this handoff")
for key in ("initContainers", "ephemeralContainers", "hostAliases", "imagePullSecrets"):
    if key in spec:
        raise SystemExit(f"{key} is not allowed")
if spec.get("restartPolicy") != "Never":
    raise SystemExit("restartPolicy must be Never")
if c.get("stdin") is not False or c.get("tty") is not False:
    raise SystemExit("interactive stdin/TTY must be disabled")
if any(key in c for key in ("args", "envFrom", "ports", "volumeMounts")):
    raise SystemExit("args, envFrom, ports, and volumeMounts are not allowed")
if c.get("imagePullPolicy") != "Always":
    raise SystemExit("imagePullPolicy must be Always")
security = c.get("securityContext", {})
required = {
    "privileged": False,
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "runAsNonRoot": True,
}
for key, value in required.items():
    if security.get(key) is not value:
        raise SystemExit(f"securityContext.{key} must be {value}")
if security.get("capabilities", {}).get("drop") != ["ALL"]:
    raise SystemExit("all Linux capabilities must be dropped")
if security.get("seccompProfile", {}).get("type") != "RuntimeDefault":
    raise SystemExit("RuntimeDefault seccomp is required")
if not isinstance(security.get("runAsUser"), int) or not isinstance(security.get("runAsGroup"), int):
    raise SystemExit("fixed numeric runAsUser and runAsGroup are required")
if c.get("resources", {}).get("limits", {}).get("nvidia.com/pgpu") != "1":
    raise SystemExit("exactly one confidential GPU is required")
if not isinstance(c.get("command"), list) or not c["command"]:
    raise SystemExit("an explicit command vector is required")
annotation = pod.get("metadata", {}).get("annotations", {}).get(
    "io.katacontainers.config.hypervisor.cc_init_data", ""
)
try:
    initdata = gzip.decompress(base64.b64decode(annotation, validate=True)).decode()
except Exception as exc:
    raise SystemExit(f"invalid embedded confidential init-data: {exc}")
for request in ("ExecProcessRequest", "ReadStreamRequest", "WriteStreamRequest", "SetPolicyRequest"):
    if not re.search(rf"default\s+{request}\s*:?=\s*false", initdata):
        raise SystemExit(f"embedded policy does not default-deny {request}")
if not re.search(r"AllowRequestsFailingPolicy[^\n]*false", initdata):
    raise SystemExit("embedded policy is not fail-closed")
for expected_fix in (
    'p_mount.source != ""',
    'p_mount.source == ""',
    'i_storage.driver in {"blk", "scsi"}',
    'expect_root_path == i_storage.mount_point',
):
    if expected_fix not in initdata:
        raise SystemExit(
            f"embedded policy lacks CVE-2026-77176 workaround: {expected_fix}"
        )
print("Static handoff invariants passed")
print("Pod:", pod["metadata"]["name"])
print("Image:", image)
print("Command:", json.dumps(c["command"]))
PY

kctl apply --dry-run=server -f "${POD_FILE}" >/dev/null
printf 'Authenticated Pod SHA-256: %s\n' "${ACTUAL_SHA256}"
printf 'Do not edit this manifest. Do not use exec, attach, cp, or port-forward.\n'
read -r -p "Type APPLY to launch the exact handoff: " CONFIRM
[[ "${CONFIRM}" == 'APPLY' ]] || die 'launch cancelled'
kctl apply -f "${POD_FILE}"
POD_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["name"])' "${POD_JSON}")"
NAMESPACE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"].get("namespace", "default"))' "${POD_JSON}")"
kctl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=10m
kctl get pod "${POD_NAME}" -n "${NAMESPACE}" -o wide
