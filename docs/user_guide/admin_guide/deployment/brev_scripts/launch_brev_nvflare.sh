#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch one prepared NVFlare participant inside a Brev single-node Kubernetes
environment. Run this script separately in the server, site-1, and site-2 Brev
environments after prepare_brev_startup_kits.sh copies the matching archive.
The Brev instance names are only needed by prepare_brev_startup_kits.sh; this
script runs inside whichever environment you selected.

Usage:
  bash launch_brev_nvflare.sh <participant-name>

Examples:
  IMAGE=registry.example.com/nvflare:dev bash launch_brev_nvflare.sh server
  IMAGE=registry.example.com/nvflare:dev SERVER_HOST=server1.example.com bash launch_brev_nvflare.sh site-1

Defaults:
  ARCHIVE=~/nvflare-<participant-name>.tgz
  WORK_DIR=~/nvflare
  NAMESPACE=nvflare
  WORKSPACE_PVC=nvflws
  DATA_PVC=nvfldata
  WORKSPACE_STORAGE=10Gi
  DATA_STORAGE=50Gi
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
  done
}

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    echo "${ROLE}"
  elif [[ -f "${PARTICIPANT_DIR}/startup/fed_server.json" ]]; then
    echo "server"
  elif [[ -f "${PARTICIPANT_DIR}/startup/fed_client.json" ]]; then
    echo "client"
  else
    fail "Cannot detect participant role from ${PARTICIPANT_DIR}/startup"
  fi
}

verify_prepared_launcher() {
  local resources_file="${PARTICIPANT_DIR}/local/resources.json.default"
  [[ -f "${resources_file}" ]] || fail "No resources.json.default found under ${PARTICIPANT_DIR}/local"
  [[ ! -f "${PARTICIPANT_DIR}/local/resources.json" ]] || {
    fail "Prepared kit contains local/resources.json. Rerun nvflare deploy prepare and copy the prepared archive."
  }

  python3 - "${resources_file}" "${ROLE}" "${NAMESPACE}" <<'PY'
import json
import sys

resources_file, role, namespace = sys.argv[1:4]

with open(resources_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if role == "server":
    expected_path = "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"
elif role == "client":
    expected_path = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
else:
    raise SystemExit(f"ROLE must be 'server' or 'client', got {role!r}")

components = data.get("components", [])
target = next((component for component in components if component.get("id") == "k8s_launcher"), None)

if target is None:
    raise SystemExit(f"No k8s_launcher component found in {resources_file}. Rerun nvflare deploy prepare.")
if target.get("path") != expected_path:
    raise SystemExit(f"k8s_launcher path is {target.get('path')!r}; expected {expected_path!r}")

args = target.get("args") or {}
prepared_namespace = args.get("namespace")
if prepared_namespace != namespace:
    raise SystemExit(
        f"Prepared launcher namespace is {prepared_namespace!r}, but launch NAMESPACE is {namespace!r}. "
        "Use the same NAMESPACE for prepare and launch."
    )
if not args.get("workspace_mount_path"):
    raise SystemExit("k8s_launcher args missing workspace_mount_path")

print(f"Verified {resources_file} contains {expected_path} for namespace {namespace}")
PY
}

write_pvc_manifest() {
  local storage_class_line=""
  if [[ -n "${STORAGE_CLASS:-}" ]]; then
    storage_class_line="  storageClassName: ${STORAGE_CLASS}"
  fi

  cat >"${PVC_FILE}" <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${WORKSPACE_PVC}
spec:
${storage_class_line}
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: ${WORKSPACE_STORAGE}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${DATA_PVC}
spec:
${storage_class_line}
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: ${DATA_STORAGE}
EOF
}

stage_workspace_pvc() {
  cat >"${COPY_POD_FILE}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${COPY_POD}
spec:
  restartPolicy: Never
  containers:
    - name: copy
      image: busybox:1.36
      command:
        - sh
        - -c
        - sleep 3600
      volumeMounts:
        - name: ${WORKSPACE_PVC}
          mountPath: /mnt/nvflws
  volumes:
    - name: ${WORKSPACE_PVC}
      persistentVolumeClaim:
        claimName: ${WORKSPACE_PVC}
EOF

  kubectl -n "${NAMESPACE}" delete pod "${COPY_POD}" --ignore-not-found=true
  kubectl -n "${NAMESPACE}" apply -f "${COPY_POD_FILE}"
  kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${COPY_POD}" --timeout=120s

  if [[ "${CLEAN_WORKSPACE_PVC:-false}" == "true" ]]; then
    kubectl -n "${NAMESPACE}" exec "${COPY_POD}" -- sh -c \
      'rm -rf /mnt/nvflws/* /mnt/nvflws/.[!.]* /mnt/nvflws/..?* 2>/dev/null || true'
  else
    kubectl -n "${NAMESPACE}" exec "${COPY_POD}" -- rm -rf /mnt/nvflws/startup /mnt/nvflws/local
  fi

  kubectl -n "${NAMESPACE}" cp "${PARTICIPANT_DIR}/startup" "${COPY_POD}:/mnt/nvflws/startup"
  kubectl -n "${NAMESPACE}" cp "${PARTICIPANT_DIR}/local" "${COPY_POD}:/mnt/nvflws/local"
  kubectl -n "${NAMESPACE}" exec "${COPY_POD}" -- ls -la /mnt/nvflws/startup /mnt/nvflws/local
  kubectl -n "${NAMESPACE}" delete pod "${COPY_POD}" --ignore-not-found=true
}

run_server_dns_check() {
  local dns_pod="${DNS_TEST_POD:-dns-test}"
  local timeout_seconds="${DNS_TEST_TIMEOUT_SECONDS:-60}"
  local phase=""
  local i

  echo "Checking DNS resolution for SERVER_HOST=${SERVER_HOST}"
  kubectl -n "${NAMESPACE}" delete pod "${dns_pod}" --ignore-not-found=true >/dev/null
  kubectl -n "${NAMESPACE}" wait --for=delete "pod/${dns_pod}" --timeout=30s >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" run "${dns_pod}" \
    --restart=Never \
    --image=busybox:1.36 \
    --command -- nslookup "${SERVER_HOST}"

  for ((i = 0; i < timeout_seconds; i++)); do
    phase="$(kubectl -n "${NAMESPACE}" get pod "${dns_pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    case "${phase}" in
      Succeeded)
        kubectl -n "${NAMESPACE}" logs "pod/${dns_pod}" || true
        kubectl -n "${NAMESPACE}" delete pod "${dns_pod}" --ignore-not-found=true >/dev/null
        return 0
        ;;
      Failed)
        kubectl -n "${NAMESPACE}" logs "pod/${dns_pod}" || true
        kubectl -n "${NAMESPACE}" describe "pod/${dns_pod}" || true
        kubectl -n "${NAMESPACE}" delete pod "${dns_pod}" --ignore-not-found=true >/dev/null
        return 1
        ;;
    esac
    sleep 1
  done

  echo "Timed out waiting for ${dns_pod} to complete. Last phase: ${phase:-unknown}" >&2
  kubectl -n "${NAMESPACE}" logs "pod/${dns_pod}" || true
  kubectl -n "${NAMESPACE}" describe "pod/${dns_pod}" || true
  kubectl -n "${NAMESPACE}" delete pod "${dns_pod}" --ignore-not-found=true >/dev/null
  return 1
}

split_image() {
  local image=$1
  local last_path_part=${image##*/}

  IMAGE_REPO="${image}"
  IMAGE_TAG=""
  if [[ "${last_path_part}" == *:* ]]; then
    IMAGE_REPO="${image%:*}"
    IMAGE_TAG="${image##*:}"
  fi
}

install_chart() {
  local helm_args=()
  helm_args=(upgrade --install "${PARTICIPANT}" "${PARTICIPANT_DIR}/helm_chart" --namespace "${NAMESPACE}")

  if [[ -n "${IMAGE:-}" ]]; then
    split_image "${IMAGE}"
    helm_args+=(--set "image.repository=${IMAGE_REPO}")
    if [[ -n "${IMAGE_TAG}" ]]; then
      helm_args+=(--set "image.tag=${IMAGE_TAG}")
    fi
  fi

  if [[ "${ROLE}" == "server" ]]; then
    helm_args+=(--set service.type=ClusterIP --set hostPortEnabled=true)
  fi

  helm "${helm_args[@]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PARTICIPANT="${1:-${PARTICIPANT:-}}"
[[ -n "${PARTICIPANT}" ]] || {
  usage
  exit 1
}

NAMESPACE="${NAMESPACE:-nvflare}"
WORKSPACE_PVC="${WORKSPACE_PVC:-nvflws}"
DATA_PVC="${DATA_PVC:-nvfldata}"
WORKSPACE_STORAGE="${WORKSPACE_STORAGE:-10Gi}"
DATA_STORAGE="${DATA_STORAGE:-50Gi}"
WORK_DIR="${WORK_DIR:-${HOME}/nvflare}"
ARCHIVE="${ARCHIVE:-${HOME}/nvflare-${PARTICIPANT}.tgz}"
COPY_POD="${COPY_POD:-nvflare-pvc-copy}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
LOG_TAIL="${LOG_TAIL:-100}"

require_cmd kubectl helm tar python3
[[ -f "${ARCHIVE}" ]] || fail "Archive not found: ${ARCHIVE}"

mkdir -p "${WORK_DIR}"
if [[ -d "${WORK_DIR}/${PARTICIPANT}" ]]; then
  backup_dir="${WORK_DIR}/${PARTICIPANT}.bak.$(date +%Y%m%d%H%M%S)"
  mv "${WORK_DIR}/${PARTICIPANT}" "${backup_dir}"
  echo "Moved existing ${WORK_DIR}/${PARTICIPANT} to ${backup_dir}"
fi

tar -xzf "${ARCHIVE}" -C "${WORK_DIR}"
PARTICIPANT_DIR="${WORK_DIR}/${PARTICIPANT}"
[[ -d "${PARTICIPANT_DIR}/helm_chart" ]] || fail "Helm chart not found: ${PARTICIPANT_DIR}/helm_chart"
[[ -d "${PARTICIPANT_DIR}/startup" ]] || fail "startup directory not found: ${PARTICIPANT_DIR}/startup"
[[ -d "${PARTICIPANT_DIR}/local" ]] || fail "local directory not found: ${PARTICIPANT_DIR}/local"
ROLE="$(detect_role)"
verify_prepared_launcher

PVC_FILE="${WORK_DIR}/nvflare-pvcs.yaml"
COPY_POD_FILE="${WORK_DIR}/copy-to-pvcs.yaml"

kubectl get nodes
helm version

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
write_pvc_manifest
kubectl -n "${NAMESPACE}" apply -f "${PVC_FILE}"
kubectl -n "${NAMESPACE}" get pvc

trap 'kubectl -n "${NAMESPACE}" delete pod "${COPY_POD}" --ignore-not-found=true >/dev/null 2>&1 || true' EXIT
stage_workspace_pvc
trap - EXIT

if [[ "${ROLE}" == "client" && -n "${SERVER_HOST:-}" ]]; then
  run_server_dns_check
fi

install_chart

kubectl -n "${NAMESPACE}" rollout status "deployment/${PARTICIPANT}" --timeout="${ROLLOUT_TIMEOUT}"
kubectl -n "${NAMESPACE}" get pods
kubectl -n "${NAMESPACE}" logs "deploy/${PARTICIPANT}" --tail="${LOG_TAIL}" || true

echo "Launched ${PARTICIPANT} (${ROLE}) in namespace ${NAMESPACE}."
