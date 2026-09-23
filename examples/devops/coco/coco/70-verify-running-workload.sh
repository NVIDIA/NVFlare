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
[[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || die 'expected SHA-256 is not 64 lowercase hex characters'
need_file "${POD_FILE}"
for command in kubectl python3 sha256sum wc; do need_cmd "${command}"; done

ACTUAL_SHA256="$(sha256sum "${POD_FILE}" | awk '{print $1}')"
[[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256}" ]] \
    || die "Pod SHA-256 mismatch"

EXPECTED_JSON="$(mktemp)"
LIVE_JSON="$(mktemp)"
LOG_OUTPUT="$(mktemp)"
LOG_ERROR="$(mktemp)"
EXEC_OUTPUT="$(mktemp)"
cleanup() {
    rm -f -- "${EXPECTED_JSON}" "${LIVE_JSON}" "${LOG_OUTPUT}" "${LOG_ERROR}" "${EXEC_OUTPUT}"
}
trap cleanup EXIT

kctl create --dry-run=client --validate=false -f "${POD_FILE}" -o json \
    > "${EXPECTED_JSON}"
mapfile -d '' -t POD_META < <(
    python3 - "${EXPECTED_JSON}" <<'PY'
import json
import sys

pod = json.load(open(sys.argv[1], encoding="utf-8"))
sys.stdout.write(pod["metadata"].get("namespace", "default") + "\0")
sys.stdout.write(pod["metadata"]["name"] + "\0")
PY
)
[[ "${#POD_META[@]}" -eq 2 ]] || die 'could not read Pod name and namespace'
NAMESPACE="${POD_META[0]}"
POD_NAME="${POD_META[1]}"

kctl wait --for=condition=Ready "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=2m
kctl get pod "${POD_NAME}" -n "${NAMESPACE}" -o json > "${LIVE_JSON}"

python3 - "${EXPECTED_JSON}" "${LIVE_JSON}" "${RUNTIME_CLASS}" <<'PY'
import json
import sys

expected = json.load(open(sys.argv[1], encoding="utf-8"))
live = json.load(open(sys.argv[2], encoding="utf-8"))
runtime = sys.argv[3]
ec = expected["spec"]["containers"][0]
lc = live["spec"]["containers"][0]
if live["spec"].get("runtimeClassName") != runtime:
    raise SystemExit("live Pod uses the wrong runtime class")
if lc.get("image") != ec.get("image") or lc.get("command") != ec.get("command"):
    raise SystemExit("live image or command differs from authenticated handoff")
if live["metadata"].get("annotations", {}).get(
    "io.katacontainers.config.hypervisor.cc_init_data"
) != expected["metadata"].get("annotations", {}).get(
    "io.katacontainers.config.hypervisor.cc_init_data"
):
    raise SystemExit("live confidential init-data differs from authenticated handoff")
statuses = live.get("status", {}).get("containerStatuses", [])
if len(statuses) != 1 or statuses[0].get("ready") is not True:
    raise SystemExit("workload container is not Ready")
if statuses[0].get("restartCount") != 0:
    raise SystemExit("workload container restarted")
if live.get("status", {}).get("phase") != "Running":
    raise SystemExit("Pod is not Running")
print("Live Pod matches authenticated image, command, init-data, and runtime")
PY

# Both the demo and provisioned NVFlare images are intentionally silent. The
# provisioning builder redirects NVFlare startup before signing the private kit.
# A single visible byte is enough to reject a release; never print its content.
set +e
kctl logs "${POD_NAME}" -n "${NAMESPACE}" --limit-bytes=1 --request-timeout=30s \
    > "${LOG_OUTPUT}" 2> "${LOG_ERROR}"
LOG_STATUS=$?
set -e
LOG_BYTES="$(wc -c < "${LOG_OUTPUT}")"
[[ "${LOG_BYTES}" -eq 0 ]] \
    || die 'silent workload emitted output visible to CoCo; owner must rebuild and authorize a new release'
if [[ "${LOG_STATUS}" -eq 0 ]]; then
    printf 'Silent workload: no application output observed through kubectl logs.\n'
else
    grep -Fq 'ReadStreamRequest is blocked by policy' "${LOG_ERROR}" \
        || die 'could not verify silent workload logs; Kubernetes access/transport errors are not a policy denial'
    printf 'Application log access denied by guest ReadStreamRequest policy.\n'
fi

set +e
kctl exec "${POD_NAME}" -n "${NAMESPACE}" -- /coco-app \
    > "${EXEC_OUTPUT}" 2>&1
EXEC_STATUS=$?
set -e
[[ "${EXEC_STATUS}" -ne 0 ]] || die 'kubectl exec unexpectedly succeeded'
grep -Fq 'ExecProcessRequest is blocked by policy' "${EXEC_OUTPUT}" \
    || die 'kubectl exec failed, but not at the expected guest-policy boundary'
printf 'kubectl exec: PermissionDenied (ExecProcessRequest is blocked by policy)\n'

kctl get pod "${POD_NAME}" -n "${NAMESPACE}" -o wide
printf 'Cluster-side checks passed for authenticated Pod %s.\n' \
    "${ACTUAL_SHA256}"
printf 'These checks do not establish NVFlare registration, peer attestation, or application readiness.\n'
printf 'The trusted federation operator must complete provision/VERIFY-RUNNING-FEDERATION.md.\n'
