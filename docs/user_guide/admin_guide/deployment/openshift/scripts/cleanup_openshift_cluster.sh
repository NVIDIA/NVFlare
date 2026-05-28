#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Delete NVFlare OpenShift scripted deployment resources and stop Red Hat
OpenShift Local (CRC).

By default this script deletes the resources created by the scripted NVFlare
workflow in NAMESPACE, then runs `crc stop`. Use --delete-namespace only when
the project is dedicated to this test; it deletes everything in that namespace.

Usage:
  bash cleanup_openshift_cluster.sh [--delete-namespace] [--no-stop] [--delete-work-dir]

Common environment:
  KUBE_CMD=oc
  HELM_BIN=helm
  CRC_BIN=crc
  NAMESPACE=nvflare-e2e
  SERVER_NAME=nvflare-server
  CLIENTS="site-1 site-2"
  ADMIN_POD=nvflare-admin
  WORK_DIR=/tmp/nvflare/openshift-e2e
  STOP_CLUSTER=true
  DELETE_NAMESPACE=false
  DELETE_WORK_DIR=false

Examples:
  bash docs/user_guide/admin_guide/deployment/openshift/scripts/cleanup_openshift_cluster.sh

  DELETE_NAMESPACE=true \
    bash docs/user_guide/admin_guide/deployment/openshift/scripts/cleanup_openshift_cluster.sh
EOF
}

info() {
  echo
  echo "==> $*"
}

warn() {
  echo "WARNING: $*" >&2
}

is_truthy() {
  case "${1:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Yy])
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || {
      warn "Required command not found: $cmd"
      return 1
    }
  done
}

safe_name() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]_' '[:lower:]-' | sed 's/[^a-z0-9-]/-/g; s/-\{1,\}/-/g; s/^-//; s/-$//' | cut -c 1-63 | sed 's/-$//')"
  if [[ -z "${name}" ]]; then
    name="site"
  fi
  case "${name}" in
    [a-z]*)
      ;;
    *)
      name="site-${name}"
      ;;
  esac
  printf '%s\n' "${name}"
}

normalize_job_id() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]//g')"
  case "${name}" in
    [0-9]*)
      name="j${name}"
      ;;
  esac
  printf '%s\n' "$(printf '%s' "${name}" | cut -c 1-63 | sed 's/-$//')"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --delete-namespace)
        DELETE_NAMESPACE=true
        ;;
      --keep-namespace)
        DELETE_NAMESPACE=false
        ;;
      --no-stop)
        STOP_CLUSTER=false
        ;;
      --delete-work-dir)
        DELETE_WORK_DIR=true
        ;;
      *)
        warn "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

namespace_exists() {
  "${KUBE_CMD}" get namespace "${NAMESPACE}" >/dev/null 2>&1
}

delete_namespace() {
  if ! namespace_exists; then
    info "Namespace ${NAMESPACE} does not exist"
    return
  fi

  info "Deleting namespace ${NAMESPACE}"
  "${KUBE_CMD}" delete namespace "${NAMESPACE}" --ignore-not-found=true
}

delete_known_job_pods() {
  local job_id_file="${WORK_DIR}/last_job_id"
  local raw_job_id=""
  local normalized_job_id=""

  if [[ ! -f "${job_id_file}" ]]; then
    return
  fi

  raw_job_id="$(sed -n '1p' "${job_id_file}")"
  [[ -n "${raw_job_id}" ]] || return
  normalized_job_id="$(normalize_job_id "${raw_job_id}")"
  "${KUBE_CMD}" -n "${NAMESPACE}" get pods -o name 2>/dev/null \
    | while IFS= read -r pod; do
        case "${pod}" in
          pod/"${normalized_job_id}"*)
            "${KUBE_CMD}" -n "${NAMESPACE}" delete "${pod}" --ignore-not-found=true
            ;;
        esac
      done
}

delete_generated_resources() {
  local participant
  local safe

  if ! namespace_exists; then
    info "Namespace ${NAMESPACE} does not exist"
    return
  fi

  info "Uninstalling Helm releases"
  for participant in ${SERVER_NAME} ${CLIENTS}; do
    "${HELM_BIN}" uninstall "${participant}" -n "${NAMESPACE}" >/dev/null 2>&1 || true
  done

  info "Deleting temporary pods and last submitted job pods"
  "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "${ADMIN_POD}" --ignore-not-found=true >/dev/null 2>&1 || true
  for participant in ${SERVER_NAME} ${CLIENTS}; do
    safe="$(safe_name "${participant}")"
    "${KUBE_CMD}" -n "${NAMESPACE}" delete pod "nvflare-copy-${safe}" --ignore-not-found=true >/dev/null 2>&1 || true
  done
  delete_known_job_pods

  info "Deleting generated workspace PVCs"
  for participant in ${SERVER_NAME} ${CLIENTS}; do
    safe="$(safe_name "${participant}")"
    "${KUBE_CMD}" -n "${NAMESPACE}" delete pvc "nvflare-ws-${safe}" --ignore-not-found=true >/dev/null 2>&1 || true
  done
}

delete_work_dir() {
  if ! is_truthy "${DELETE_WORK_DIR}"; then
    return
  fi

  case "${WORK_DIR}" in
    /tmp/nvflare/*)
      info "Deleting work directory ${WORK_DIR}"
      rm -rf "${WORK_DIR}"
      ;;
    *)
      warn "Refusing to delete WORK_DIR outside /tmp/nvflare: ${WORK_DIR}"
      ;;
  esac
}

stop_cluster() {
  if ! is_truthy "${STOP_CLUSTER}"; then
    return
  fi

  if ! command -v "${CRC_BIN}" >/dev/null 2>&1; then
    warn "CRC command not found: ${CRC_BIN}; skipping cluster stop"
    return
  fi

  info "Stopping OpenShift Local"
  "${CRC_BIN}" stop
}

KUBE_CMD="${KUBE_CMD:-oc}"
HELM_BIN="${HELM_BIN:-helm}"
CRC_BIN="${CRC_BIN:-crc}"
NAMESPACE="${NAMESPACE:-nvflare-e2e}"
SERVER_NAME="${SERVER_NAME:-nvflare-server}"
CLIENTS="${CLIENTS:-site-1 site-2}"
ADMIN_POD="${ADMIN_POD:-nvflare-admin}"
WORK_DIR="${WORK_DIR:-/tmp/nvflare/openshift-e2e}"
STOP_CLUSTER="${STOP_CLUSTER:-true}"
DELETE_NAMESPACE="${DELETE_NAMESPACE:-false}"
DELETE_WORK_DIR="${DELETE_WORK_DIR:-false}"

parse_args "$@"

if is_truthy "${DELETE_NAMESPACE}"; then
  if require_cmd "${KUBE_CMD}"; then
    delete_namespace
  else
    warn "Skipping namespace cleanup because the Kubernetes CLI is missing"
  fi
else
  if require_cmd "${KUBE_CMD}" "${HELM_BIN}"; then
    delete_generated_resources
  else
    warn "Skipping resource cleanup because required Kubernetes tools are missing"
  fi
fi

delete_work_dir
stop_cluster
