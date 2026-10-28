#!/usr/bin/env bash

# GPU Operator invariants required by the confidential Kata runtime. The caller
# supplies kctl, die, GPU_RESOURCE, and GPU_OPERATOR_DRIVER_ENABLED.

gpu_operator_driver_value() {
  case "${GPU_OPERATOR_DRIVER_ENABLED:-1}" in
    1) printf 'true\n' ;;
    0) printf 'false\n' ;;
    *) die 'GPU_OPERATOR_DRIVER_ENABLED must be 0 or 1' ;;
  esac
}

load_gpu_operator_expectations() {
  local selected_runtime="${RUNTIME_CLASS}" value configured_runtime configured_resource
  [[ -s "${COCO_CONFIG}" ]] || die "Missing CoCo install configuration: ${COCO_CONFIG}"
  value="$({ source "${COCO_CONFIG}"; printf '%s' "${GPU_OPERATOR_DRIVER_ENABLED:-1}"; })"
  [[ "${value}" =~ ^[01]$ ]] || die 'GPU_OPERATOR_DRIVER_ENABLED must be 0 or 1 in COCO_CONFIG'
  configured_runtime="$({ source "${COCO_CONFIG}"; printf '%s' "${RUNTIME_CLASS:-}"; })"
  [[ "${configured_runtime}" == "${selected_runtime}" ]] ||
    die 'RUNTIME_CLASS differs between platform.env and COCO_CONFIG'
  configured_resource="$({ source "${COCO_CONFIG}"; printf '%s' "${GPU_RESOURCE:-}"; })"
  [[ -n "${configured_resource}" ]] || die 'GPU_RESOURCE is missing from COCO_CONFIG'
  GPU_OPERATOR_DRIVER_ENABLED="${value}"
  GPU_RESOURCE="${configured_resource}"
  export GPU_OPERATOR_DRIVER_ENABLED GPU_RESOURCE
}

gpu_operator_operand_report() {
  local policy driver
  driver="$(gpu_operator_driver_value)"
  if ! policy="$(kctl get clusterpolicy cluster-policy -o json 2>/dev/null)"; then
    printf 'Expected ClusterPolicy cluster-policy, but it is absent or unreadable.\n' >&2
    return 1
  fi
  python3 - "${driver}" 3<<<"${policy}" <<'PY'
import json
import os
import sys

expected_driver = sys.argv[1] == "true"
try:
    with os.fdopen(3) as stream:
        policy = json.load(stream)
except (OSError, TypeError, ValueError):
    print("ClusterPolicy cluster-policy did not return valid JSON", file=sys.stderr)
    raise SystemExit(1)

spec = policy.get("spec", {})
expected = {
    "sandboxWorkloads.enabled": True,
    "sandboxWorkloads.mode": "kata",
    "sandboxWorkloads.defaultWorkload": "container",
    "ccManager.enabled": True,
    "ccManager.defaultMode": "on",
    "vfioManager.enabled": True,
    "kataSandboxDevicePlugin.enabled": True,
    "kataManager.enabled": False,
    "driver.enabled": expected_driver,
}

def lookup(path):
    value = spec
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return "<missing>"
        value = value[part]
    return value

mismatches = [(path, value, lookup(path)) for path, value in expected.items() if lookup(path) != value]
for path, expected_value, actual in mismatches:
    print(f"ClusterPolicy operand {path}: expected {expected_value!r}, actual {actual!r}", file=sys.stderr)
raise SystemExit(bool(mismatches))
PY
}

gpu_operator_node_status() {
  local node_name=$1 escaped_resource label allocatable
  label="$(kctl get node "${node_name}" -o jsonpath='{.metadata.labels.nvidia\.com/cc\.ready\.state}' 2>/dev/null || true)"
  escaped_resource="${GPU_RESOURCE//./\\.}"
  allocatable="$(kctl get node "${node_name}" -o "jsonpath={.status.allocatable.${escaped_resource}}" 2>/dev/null || true)"
  [[ "${label}" == true && "${allocatable:-0}" != 0 ]]
}

report_gpu_operator_node_status() {
  local node_name=$1 escaped_resource label allocatable
  label="$(kctl get node "${node_name}" -o jsonpath='{.metadata.labels.nvidia\.com/cc\.ready\.state}' 2>/dev/null || true)"
  escaped_resource="${GPU_RESOURCE//./\\.}"
  allocatable="$(kctl get node "${node_name}" -o "jsonpath={.status.allocatable.${escaped_resource}}" 2>/dev/null || true)"
  printf 'Node %s label nvidia.com/cc.ready.state: expected true, actual %s\n' \
    "${node_name}" "${label:-<missing>}" >&2
  printf 'Node %s allocatable %s: expected nonzero, actual %s\n' \
    "${node_name}" "${GPU_RESOURCE}" "${allocatable:-<missing>}" >&2
}

assert_gpu_operator_readiness() {
  local node_name=${1:-}
  [[ -n "${node_name}" ]] || node_name="$(kctl get nodes -o jsonpath='{.items[0].metadata.name}')"
  [[ -n "${node_name}" ]] || die 'No Kubernetes node found'
  gpu_operator_operand_report || die 'GPU Operator ClusterPolicy drifted; restore the documented CoCo values'
  if ! gpu_operator_node_status "${node_name}"; then
    report_gpu_operator_node_status "${node_name}"
    die 'GPU Operator is not ready for confidential Kata passthrough'
  fi
}

wait_for_gpu_operator_readiness() {
  local node_name=$1 timeout=$2 deadline=$((SECONDS + timeout))
  until gpu_operator_operand_report >/dev/null 2>&1 && gpu_operator_node_status "${node_name}"; do
    if ((SECONDS >= deadline)); then
      gpu_operator_operand_report || true
      report_gpu_operator_node_status "${node_name}"
      die 'Timed out waiting for GPU Operator confidential-computing readiness'
    fi
    sleep 10
  done
}
