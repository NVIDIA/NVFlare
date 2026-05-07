#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch one provisioned NVFlare participant inside a Brev single-node Kubernetes
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
  WORKSPACE_MOUNT_PATH=/var/tmp/nvflare/workspace
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

infer_role() {
  if [[ -n "${ROLE:-}" ]]; then
    echo "${ROLE}"
  elif [[ "${PARTICIPANT}" == "server" || "${PARTICIPANT}" == "server1" ]]; then
    echo "server"
  else
    echo "client"
  fi
}

patch_launcher() {
  local resources_file
  if [[ -f "${PARTICIPANT_DIR}/local/resources.json" ]]; then
    resources_file="${PARTICIPANT_DIR}/local/resources.json"
  elif [[ -f "${PARTICIPANT_DIR}/local/resources.json.default" ]]; then
    resources_file="${PARTICIPANT_DIR}/local/resources.json.default"
  else
    fail "No resources.json or resources.json.default found under ${PARTICIPANT_DIR}/local"
  fi

  python3 - "${resources_file}" "${ROLE}" "${NAMESPACE}" "${WORKSPACE_MOUNT_PATH}" <<'PY'
import json
import sys

resources_file, role, namespace, workspace_mount_path = sys.argv[1:5]

with open(resources_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if role == "server":
    old_path = "nvflare.app_common.job_launcher.server_process_launcher.ServerProcessJobLauncher"
    new_path = "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"
elif role == "client":
    old_path = "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher"
    new_path = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
else:
    raise SystemExit(f"ROLE must be 'server' or 'client', got {role!r}")

components = data.get("components", [])
target = None
for component in components:
    if component.get("path") in {old_path, new_path} or component.get("id") in {"process_launcher", "k8s_launcher"}:
        target = component
        break

if target is None:
    raise SystemExit(f"No process launcher component found in {resources_file}")

target["id"] = "k8s_launcher"
target["path"] = new_path
target["args"] = {
    "config_file_path": None,
    "study_data_pvc_file_path": f"{workspace_mount_path}/local/study_data.yaml",
    "namespace": namespace,
    "python_path": "/usr/local/bin/python3",
    "workspace_mount_path": workspace_mount_path,
    "pending_timeout": 300,
    "ephemeral_storage": "1Gi",
}

with open(resources_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Patched {resources_file} to use {new_path}")
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
  fi

  kubectl -n "${NAMESPACE}" cp "${PARTICIPANT_DIR}/." "${COPY_POD}:/mnt/nvflws/"
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
WORKSPACE_MOUNT_PATH="${WORKSPACE_MOUNT_PATH:-/var/tmp/nvflare/workspace}"
WORK_DIR="${WORK_DIR:-${HOME}/nvflare}"
ARCHIVE="${ARCHIVE:-${HOME}/nvflare-${PARTICIPANT}.tgz}"
COPY_POD="${COPY_POD:-nvflare-pvc-copy}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
LOG_TAIL="${LOG_TAIL:-100}"
ROLE="$(infer_role)"

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

PVC_FILE="${WORK_DIR}/nvflare-pvcs.yaml"
COPY_POD_FILE="${WORK_DIR}/copy-to-pvcs.yaml"

kubectl get nodes
helm version

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
write_pvc_manifest
kubectl -n "${NAMESPACE}" apply -f "${PVC_FILE}"
kubectl -n "${NAMESPACE}" get pvc

patch_launcher

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
