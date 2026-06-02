#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Configure and create a local single-node OpenShift cluster for NVFlare testing
with Red Hat OpenShift Local (CRC).

OpenShift Local creates the cluster VM during the first `crc start`. This script
sets the CRC configuration, runs `crc setup`, and starts the cluster by default
through start_openshift_cluster.sh.

Usage:
  bash create_openshift_cluster.sh [--setup-only]

Required for the first non-interactive start:
  PULL_SECRET_FILE  Path to the OpenShift pull secret downloaded from
                    https://console.redhat.com/openshift/create/local

Common environment:
  CRC_BIN=crc
  CRC_PRESET=openshift
  CRC_CPUS=6
  CRC_MEMORY=24576
  CRC_DISK_SIZE=120
  CRC_ENABLE_CLUSTER_MONITORING=false
  CRC_BUNDLE=<optional local or remote CRC bundle>
  CRC_HTTP_PROXY=<optional proxy URL>
  CRC_HTTPS_PROXY=<optional proxy URL>
  CRC_NO_PROXY=<optional no-proxy list>
  CRC_ENABLE_SHARED_DIRS=false
  START_AFTER_CREATE=true

Environment passed to start_openshift_cluster.sh:
  OC_BIN=oc
  NAMESPACE=nvflare-e2e
  CREATE_PROJECT=true
  LOGIN_OPENSHIFT=true
  OPENSHIFT_USER=developer
  OPENSHIFT_PASSWORD=developer
  OPENSHIFT_API_URL=https://api.crc.testing:6443

Examples:
  PULL_SECRET_FILE=$HOME/Downloads/pull-secret.txt \
    bash examples/devops/openshift/scripts/create_openshift_cluster.sh

  CRC_CPUS=8 CRC_MEMORY=32768 CRC_DISK_SIZE=160 \
  PULL_SECRET_FILE=$HOME/Downloads/pull-secret.txt \
    bash examples/devops/openshift/scripts/create_openshift_cluster.sh

  START_AFTER_CREATE=false \
    bash examples/devops/openshift/scripts/create_openshift_cluster.sh
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
      --setup-only)
        START_AFTER_CREATE=false
        ;;
      *)
        fail "Unknown argument: $1"
        ;;
    esac
    shift
  done
}

crc_config_set() {
  local key=$1
  local value=$2

  [[ -n "${value}" ]] || return 0
  info "Setting crc config ${key}=${value}"
  "${CRC_BIN}" config set "${key}" "${value}"
}

validate_pull_secret_for_noninteractive_start() {
  if ! is_truthy "${START_AFTER_CREATE}"; then
    return
  fi

  if [[ -n "${PULL_SECRET_FILE}" ]]; then
    [[ -r "${PULL_SECRET_FILE}" ]] || fail "PULL_SECRET_FILE is not readable: ${PULL_SECRET_FILE}"
    return
  fi

  [[ -t 0 ]] || fail "Set PULL_SECRET_FILE for non-interactive cluster creation, or set START_AFTER_CREATE=false."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CRC_BIN="${CRC_BIN:-crc}"
CRC_PRESET="${CRC_PRESET:-openshift}"
CRC_CPUS="${CRC_CPUS:-6}"
CRC_MEMORY="${CRC_MEMORY:-24576}"
CRC_DISK_SIZE="${CRC_DISK_SIZE:-120}"
CRC_ENABLE_CLUSTER_MONITORING="${CRC_ENABLE_CLUSTER_MONITORING:-false}"
CRC_BUNDLE="${CRC_BUNDLE:-}"
CRC_HTTP_PROXY="${CRC_HTTP_PROXY:-}"
CRC_HTTPS_PROXY="${CRC_HTTPS_PROXY:-}"
CRC_NO_PROXY="${CRC_NO_PROXY:-}"
CRC_ENABLE_SHARED_DIRS="${CRC_ENABLE_SHARED_DIRS:-false}"
PULL_SECRET_FILE="${PULL_SECRET_FILE:-}"
START_AFTER_CREATE="${START_AFTER_CREATE:-true}"

parse_args "$@"
require_cmd "${CRC_BIN}"
validate_pull_secret_for_noninteractive_start

if [[ -n "${CRC_HOME_DIR:-}" ]]; then
  info "Ignoring CRC_HOME_DIR=${CRC_HOME_DIR}; use CRC_ENABLE_SHARED_DIRS=false for nonstandard home paths."
fi

info "Configuring OpenShift Local"
crc_config_set preset "${CRC_PRESET}"
crc_config_set cpus "${CRC_CPUS}"
crc_config_set memory "${CRC_MEMORY}"
crc_config_set disk-size "${CRC_DISK_SIZE}"
crc_config_set enable-cluster-monitoring "${CRC_ENABLE_CLUSTER_MONITORING}"
crc_config_set bundle "${CRC_BUNDLE}"
crc_config_set http-proxy "${CRC_HTTP_PROXY}"
crc_config_set https-proxy "${CRC_HTTPS_PROXY}"
crc_config_set no-proxy "${CRC_NO_PROXY}"
crc_config_set enable-shared-dirs "${CRC_ENABLE_SHARED_DIRS}"

info "Running crc setup"
"${CRC_BIN}" setup

if is_truthy "${START_AFTER_CREATE}"; then
  info "Starting OpenShift Local"
  bash "${SCRIPT_DIR}/start_openshift_cluster.sh"
else
  cat <<EOF

CRC setup is complete.

Start the cluster later with:
  PULL_SECRET_FILE=${PULL_SECRET_FILE:-/path/to/pull-secret.txt} \\
    bash ${SCRIPT_DIR}/start_openshift_cluster.sh
EOF
fi
