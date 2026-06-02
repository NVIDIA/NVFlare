#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start a Red Hat OpenShift Local (CRC) cluster and prepare a namespace for the
NVFlare OpenShift deployment examples.

Usage:
  bash start_openshift_cluster.sh [--no-login] [--no-project]

Common environment:
  CRC_BIN=crc
  OC_BIN=oc
  PULL_SECRET_FILE=<path to OpenShift pull secret, required by first crc start>
  CRC_DISABLE_UPDATE_CHECK=true
  CRC_BUNDLE=<optional local or remote CRC bundle>
  CRC_NAMESERVER=<optional IPv4 DNS server>
  CRC_ENABLE_SHARED_DIRS=false
  CRC_OPENSHIFT_READY_TIMEOUT=900
  CRC_OPENSHIFT_READY_INTERVAL=10

Optional crc start sizing overrides:
  CRC_CPUS=<vCPU count>
  CRC_MEMORY=<memory in MiB>
  CRC_DISK_SIZE=<disk in GiB>

OpenShift login and project environment:
  LOGIN_OPENSHIFT=true
  OPENSHIFT_API_URL=https://api.crc.testing:6443
  OPENSHIFT_USER=developer
  OPENSHIFT_PASSWORD=developer
  OPENSHIFT_INSECURE_TLS=true
  OPENSHIFT_LOGIN_RETRIES=24
  OPENSHIFT_LOGIN_INTERVAL=5
  CREATE_PROJECT=true
  NAMESPACE=nvflare-e2e

Examples:
  PULL_SECRET_FILE=$HOME/Downloads/pull-secret.txt \
    bash examples/devops/openshift/scripts/start_openshift_cluster.sh

  OPENSHIFT_USER=kubeadmin OPENSHIFT_PASSWORD=<password> \
    bash examples/devops/openshift/scripts/start_openshift_cluster.sh

  LOGIN_OPENSHIFT=false \
    bash examples/devops/openshift/scripts/start_openshift_cluster.sh
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo
  echo "==> $*"
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
  done
}

crc_log_path() {
  printf "%s/.crc/crc.log" "${HOME}"
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

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --no-login)
        LOGIN_OPENSHIFT=false
        ;;
      --no-project)
        CREATE_PROJECT=false
        ;;
      *)
        fail "Unknown argument: $1"
        ;;
    esac
    shift
  done
}

crc_status_is_running() {
  local status

  status="$("${CRC_BIN}" status 2>/dev/null || true)"
  grep -q "OpenShift:.*Running" <<<"${status}"
}

wait_for_openshift_running() {
  local elapsed=0
  local status=""

  info "Waiting for OpenShift to report Running"
  while ((elapsed <= CRC_OPENSHIFT_READY_TIMEOUT)); do
    status="$("${CRC_BIN}" status 2>/dev/null || true)"
    if grep -q "OpenShift:.*Running" <<<"${status}"; then
      echo "${status}"
      return
    fi

    echo "${status}"
    echo "OpenShift is not ready; checking again in ${CRC_OPENSHIFT_READY_INTERVAL}s."
    sleep "${CRC_OPENSHIFT_READY_INTERVAL}"
    elapsed=$((elapsed + CRC_OPENSHIFT_READY_INTERVAL))
  done

  fail "OpenShift did not report Running within ${CRC_OPENSHIFT_READY_TIMEOUT}s. Check $(crc_log_path) or rerun '${CRC_BIN} start --log-level debug' for CRC diagnostics."
}

start_crc() {
  local start_cmd=("${CRC_BIN}" start)

  if [[ -n "${PULL_SECRET_FILE}" ]]; then
    [[ -r "${PULL_SECRET_FILE}" ]] || fail "PULL_SECRET_FILE is not readable: ${PULL_SECRET_FILE}"
    start_cmd+=("--pull-secret-file" "${PULL_SECRET_FILE}")
  fi
  if is_truthy "${CRC_DISABLE_UPDATE_CHECK}"; then
    start_cmd+=("--disable-update-check")
  fi
  [[ -n "${CRC_BUNDLE}" ]] && start_cmd+=("--bundle" "${CRC_BUNDLE}")
  [[ -n "${CRC_NAMESERVER}" ]] && start_cmd+=("--nameserver" "${CRC_NAMESERVER}")
  [[ -n "${CRC_CPUS}" ]] && start_cmd+=("--cpus" "${CRC_CPUS}")
  [[ -n "${CRC_MEMORY}" ]] && start_cmd+=("--memory" "${CRC_MEMORY}")
  [[ -n "${CRC_DISK_SIZE}" ]] && start_cmd+=("--disk-size" "${CRC_DISK_SIZE}")

  info "Running ${start_cmd[*]}"
  "${start_cmd[@]}"
}

configure_oc() {
  if command -v "${OC_BIN}" >/dev/null 2>&1; then
    return
  fi

  info "Adding CRC oc binary to PATH"
  eval "$("${CRC_BIN}" oc-env)"
  command -v "${OC_BIN}" >/dev/null 2>&1 || fail "Required command not found after crc oc-env: ${OC_BIN}"
}

login_openshift() {
  local login_cmd=("${OC_BIN}" login "${OPENSHIFT_API_URL}" "-u" "${OPENSHIFT_USER}" "-p" "${OPENSHIFT_PASSWORD}")
  local attempt
  local output

  if is_truthy "${OPENSHIFT_INSECURE_TLS}"; then
    login_cmd+=("--insecure-skip-tls-verify=true")
  fi

  info "Logging in to ${OPENSHIFT_API_URL} as ${OPENSHIFT_USER}"
  for ((attempt = 1; attempt <= OPENSHIFT_LOGIN_RETRIES; attempt++)); do
    if output="$("${login_cmd[@]}" 2>&1)"; then
      echo "${output}"
      return
    fi
    echo "Login attempt ${attempt}/${OPENSHIFT_LOGIN_RETRIES} failed:"
    echo "${output}"
    echo "Retrying in ${OPENSHIFT_LOGIN_INTERVAL}s."
    sleep "${OPENSHIFT_LOGIN_INTERVAL}"
  done

  echo "${output:-}"
  fail "Unable to log in to OpenShift. Run '${CRC_BIN} console --credentials' if you need the kubeadmin password."
}

ensure_project() {
  if ! is_truthy "${CREATE_PROJECT}"; then
    return
  fi

  info "Ensuring project ${NAMESPACE}"
  if "${OC_BIN}" get project "${NAMESPACE}" >/dev/null 2>&1; then
    "${OC_BIN}" project "${NAMESPACE}" >/dev/null
  else
    "${OC_BIN}" new-project "${NAMESPACE}" >/dev/null
  fi
}

print_summary() {
  info "Cluster status"
  "${CRC_BIN}" status || true

  if is_truthy "${LOGIN_OPENSHIFT}"; then
    info "OpenShift context"
    "${OC_BIN}" whoami || true
    "${OC_BIN}" project -q || true

    info "Storage classes"
    "${OC_BIN}" get storageclass || true
  fi

  info "Console"
  "${CRC_BIN}" console --url || true

  cat <<EOF

To use oc in a new shell:
  eval "\$(${CRC_BIN} oc-env)"

To view the kubeadmin password:
  ${CRC_BIN} console --credentials
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CRC_BIN="${CRC_BIN:-crc}"
OC_BIN="${OC_BIN:-oc}"
PULL_SECRET_FILE="${PULL_SECRET_FILE:-}"
CRC_DISABLE_UPDATE_CHECK="${CRC_DISABLE_UPDATE_CHECK:-true}"
CRC_BUNDLE="${CRC_BUNDLE:-}"
CRC_NAMESERVER="${CRC_NAMESERVER:-}"
CRC_ENABLE_SHARED_DIRS="${CRC_ENABLE_SHARED_DIRS:-false}"
CRC_OPENSHIFT_READY_TIMEOUT="${CRC_OPENSHIFT_READY_TIMEOUT:-900}"
CRC_OPENSHIFT_READY_INTERVAL="${CRC_OPENSHIFT_READY_INTERVAL:-10}"
CRC_CPUS="${CRC_CPUS:-}"
CRC_MEMORY="${CRC_MEMORY:-}"
CRC_DISK_SIZE="${CRC_DISK_SIZE:-}"
LOGIN_OPENSHIFT="${LOGIN_OPENSHIFT:-true}"
OPENSHIFT_API_URL="${OPENSHIFT_API_URL:-https://api.crc.testing:6443}"
OPENSHIFT_USER="${OPENSHIFT_USER:-developer}"
OPENSHIFT_PASSWORD="${OPENSHIFT_PASSWORD:-developer}"
OPENSHIFT_INSECURE_TLS="${OPENSHIFT_INSECURE_TLS:-true}"
OPENSHIFT_LOGIN_RETRIES="${OPENSHIFT_LOGIN_RETRIES:-24}"
OPENSHIFT_LOGIN_INTERVAL="${OPENSHIFT_LOGIN_INTERVAL:-5}"
CREATE_PROJECT="${CREATE_PROJECT:-true}"
NAMESPACE="${NAMESPACE:-nvflare-e2e}"

parse_args "$@"
require_cmd "${CRC_BIN}"

if [[ -n "${CRC_HOME_DIR:-}" ]]; then
  info "Ignoring CRC_HOME_DIR=${CRC_HOME_DIR}; use CRC_ENABLE_SHARED_DIRS=false for nonstandard home paths."
fi

info "Setting crc config enable-shared-dirs=${CRC_ENABLE_SHARED_DIRS}"
"${CRC_BIN}" config set enable-shared-dirs "${CRC_ENABLE_SHARED_DIRS}"

if crc_status_is_running; then
  info "OpenShift Local is already running"
else
  start_crc
fi

wait_for_openshift_running
configure_oc

if is_truthy "${LOGIN_OPENSHIFT}"; then
  login_openshift
  ensure_project
fi

print_summary
