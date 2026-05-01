#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate NVFlare startup kits for three Brev Kubernetes environments and copy
the kits to the matching environments with the Brev CLI.

Usage:
  bash prepare_brev_startup_kits.sh [--prompt-brev-names]

Required environment:
  SERVER_HOST  External DNS name or IP that site clusters use for the server.
               This value is written into project.yml and server certificates.
  IMAGE        Container image that all clusters can pull, for example
               registry.example.com/nvflare:dev.

Brev environment selection:
  Set SERVER_BREV, SITE_1_BREV, and SITE_2_BREV, or pass
  --prompt-brev-names to enter the Brev environment names interactively.

Defaults:
  SERVER_BREV=server
  SITE_1_BREV=site-1
  SITE_2_BREV=site-2
  SERVER_PARTICIPANT=server
  SITE_1_PARTICIPANT=site-1
  SITE_2_PARTICIPANT=site-2
  FED_LEARN_PORT=8002
  PROJECT_NAME=brev_nvflare_project
  PROVISION_WORKSPACE=/tmp/nvflare/brev-provision
  KIT_DIR=/tmp/nvflare/brev-kits

Example:
  SERVER_HOST=server1.example.com \
  IMAGE=registry.example.com/nvflare:dev \
  bash prepare_brev_startup_kits.sh

  SERVER_HOST=server1.example.com \
  IMAGE=registry.example.com/nvflare:dev \
  SERVER_BREV=my-server SITE_1_BREV=my-site-a SITE_2_BREV=my-site-b \
  bash prepare_brev_startup_kits.sh

  SERVER_HOST=server1.example.com \
  IMAGE=registry.example.com/nvflare:dev \
  bash prepare_brev_startup_kits.sh --prompt-brev-names
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

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --prompt|--prompt-brev-names)
        PROMPT_BREV_NAMES=true
        ;;
      --no-prompt|--no-prompt-brev-names)
        PROMPT_BREV_NAMES=false
        ;;
      *)
        fail "Unknown argument: $1"
        ;;
    esac
    shift
  done
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

prompt_value() {
  local __resultvar=$1
  local prompt_text=$2
  local default_value=$3
  local value

  [[ -t 0 ]] || fail "Cannot prompt without a TTY. Set SERVER_BREV, SITE_1_BREV, and SITE_2_BREV instead."

  read -r -p "${prompt_text} [${default_value}]: " value
  value="${value:-${default_value}}"
  printf -v "${__resultvar}" '%s' "${value}"
}

configure_brev_instance_names() {
  SERVER_BREV="${SERVER_BREV:-server}"
  SITE_1_BREV="${SITE_1_BREV:-site-1}"
  SITE_2_BREV="${SITE_2_BREV:-site-2}"

  if is_truthy "${PROMPT_BREV_NAMES:-false}"; then
    prompt_value SERVER_BREV "Brev environment for ${SERVER_PARTICIPANT}" "${SERVER_BREV}"
    prompt_value SITE_1_BREV "Brev environment for ${SITE_1_PARTICIPANT}" "${SITE_1_BREV}"
    prompt_value SITE_2_BREV "Brev environment for ${SITE_2_PARTICIPANT}" "${SITE_2_BREV}"
  fi

  [[ -n "${SERVER_BREV}" ]] || fail "SERVER_BREV cannot be empty"
  [[ -n "${SITE_1_BREV}" ]] || fail "SITE_1_BREV cannot be empty"
  [[ -n "${SITE_2_BREV}" ]] || fail "SITE_2_BREV cannot be empty"
}

write_project_file() {
  cat >"${PROJECT_FILE}" <<EOF
api_version: 3
name: ${PROJECT_NAME}
description: NVFlare Brev Kubernetes Helm deployment for three Brev environments

participants:
  - name: ${SERVER_PARTICIPANT}
    type: server
    org: ${ORG}
    default_host: "${SERVER_HOST}"
    host_names:
      - "${SERVER_PARTICIPANT}"
      - "${SERVER_HOST}"
    fed_learn_port: ${FED_LEARN_PORT}
  - name: ${SITE_1_PARTICIPANT}
    type: client
    org: ${ORG}
  - name: ${SITE_2_PARTICIPANT}
    type: client
    org: ${ORG}
  - name: ${ADMIN_USER}
    type: admin
    org: ${ORG}
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file:
        - master_template.yml
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      scheme: tcp
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
EOF
}

write_prepare_config() {
  cat >"${PREPARE_CONFIG}" <<EOF
runtime: k8s
parent:
  docker_image: "${IMAGE}"
  parent_port: ${PARENT_PORT}
  workspace_pvc: ${WORKSPACE_PVC}
  workspace_mount_path: ${WORKSPACE_MOUNT_PATH}
job_launcher:
  config_file_path:
  default_python_path: /usr/local/bin/python3
  pending_timeout: 300
EOF
}

project_name_from_file() {
  awk -F: '
    $1 == "name" {
      value = $2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "$1"
}

latest_prod_dir() {
  local parent=$1
  local dirs=()
  shopt -s nullglob
  dirs=("${parent}"/prod_*)
  shopt -u nullglob
  if ((${#dirs[@]} == 0)); then
    return 1
  fi
  printf '%s\n' "${dirs[@]}" | sort | tail -n 1
}

copy_participant() {
  local participant=$1
  local brev_env=$2
  local archive="${KIT_DIR}/nvflare-${participant}.tgz"
  local participant_dir="${PREPARED_DIR}/${participant}"

  [[ -d "${participant_dir}" ]] || fail "Prepared participant folder not found: ${participant_dir}"
  [[ -d "${participant_dir}/helm_chart" ]] || fail "Helm chart not found for participant: ${participant}"

  tar -czf "${archive}" -C "${PREPARED_DIR}" "${participant}"
  brev copy "${archive}" "${brev_env}:${REMOTE_DIR}/"
  brev copy "${LAUNCH_SCRIPT}" "${brev_env}:${REMOTE_DIR}/"

  echo "Copied ${archive} and $(basename "${LAUNCH_SCRIPT}") to ${brev_env}:${REMOTE_DIR}/"
}

prepare_participant() {
  local participant=$1
  local source_dir="${PROD_DIR}/${participant}"
  local output_dir="${PREPARED_DIR}/${participant}"

  [[ -d "${source_dir}" ]] || fail "Provisioned participant folder not found: ${source_dir}"
  nvflare deploy prepare "${source_dir}" --output "${output_dir}" --config "${PREPARE_CONFIG}"
}

report_port_exposure() {
  if [[ "${REPORT_SERVER_PORT_EXPOSURE:-true}" != "true" ]]; then
    return
  fi

  cat <<EOF

Server port exposure:
  The Brev CLI installed in this environment does not provide a public TCP
  expose command. It provides 'brev port-forward', but that is local-only and
  is not sufficient for remote site clusters to reach the FL server.

  In the Brev UI, expose TCP port ${FED_LEARN_PORT} for environment '${SERVER_BREV}'.
  SERVER_HOST must resolve to the exposed host/IP, without the port suffix.
EOF
}

PROMPT_BREV_NAMES="${PROMPT_BREV_NAMES:-false}"
parse_args "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${SCRIPT_DIR}/launch_brev_nvflare.sh}"

PROJECT_NAME="${PROJECT_NAME:-brev_nvflare_project}"
SERVER_PARTICIPANT="${SERVER_PARTICIPANT:-server}"
SITE_1_PARTICIPANT="${SITE_1_PARTICIPANT:-site-1}"
SITE_2_PARTICIPANT="${SITE_2_PARTICIPANT:-site-2}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu}"
PROVISION_WORKSPACE="${PROVISION_WORKSPACE:-/tmp/nvflare/brev-provision}"
KIT_DIR="${KIT_DIR:-/tmp/nvflare/brev-kits}"
PREPARED_DIR="${PREPARED_DIR:-${KIT_DIR}/prepared}"
PREPARE_CONFIG="${PREPARE_CONFIG:-${KIT_DIR}/deploy-prepare-k8s.yaml}"
FED_LEARN_PORT="${FED_LEARN_PORT:-8002}"
PARENT_PORT="${PARENT_PORT:-8102}"
WORKSPACE_PVC="${WORKSPACE_PVC:-nvflws}"
WORKSPACE_MOUNT_PATH="${WORKSPACE_MOUNT_PATH:-/var/tmp/nvflare/workspace}"
ORG="${ORG:-nvidia}"
ADMIN_USER="${ADMIN_USER:-admin@nvidia.com}"
PROJECT_FILE_WAS_SET=false
if [[ -n "${PROJECT_FILE:-}" ]]; then
  PROJECT_FILE_WAS_SET=true
else
  PROJECT_FILE="${KIT_DIR}/project.yml"
fi

: "${SERVER_HOST:?Set SERVER_HOST to the external server DNS name or IP before provisioning.}"
: "${IMAGE:?Set IMAGE to a container image that all clusters can pull.}"

require_cmd nvflare brev tar awk sort tail
[[ -f "${LAUNCH_SCRIPT}" ]] || fail "Launch script not found: ${LAUNCH_SCRIPT}"

configure_brev_instance_names

mkdir -p "${KIT_DIR}" "${PROVISION_WORKSPACE}"

if [[ "${GENERATE_PROJECT_FILE:-}" == "true" || "${PROJECT_FILE_WAS_SET}" == "false" || ! -f "${PROJECT_FILE}" ]]; then
  write_project_file
  echo "Wrote generated project file: ${PROJECT_FILE}"
else
  echo "Using existing project file: ${PROJECT_FILE}"
fi

PROJECT_NAME_IN_FILE="$(project_name_from_file "${PROJECT_FILE}")"
[[ -n "${PROJECT_NAME_IN_FILE}" ]] || fail "Unable to read top-level 'name:' from ${PROJECT_FILE}"

echo "Running nvflare provision..."
nvflare provision -p "${PROJECT_FILE}" -w "${PROVISION_WORKSPACE}"

PROD_PARENT="${PROVISION_WORKSPACE}/${PROJECT_NAME_IN_FILE}"
PROD_DIR="$(latest_prod_dir "${PROD_PARENT}")" || fail "No prod_* folder found under ${PROD_PARENT}"
echo "Using provisioned production folder: ${PROD_DIR}"

rm -rf "${PREPARED_DIR}"
mkdir -p "${PREPARED_DIR}"
write_prepare_config
prepare_participant "${SERVER_PARTICIPANT}"
prepare_participant "${SITE_1_PARTICIPANT}"
prepare_participant "${SITE_2_PARTICIPANT}"

copy_participant "${SERVER_PARTICIPANT}" "${SERVER_BREV}"
copy_participant "${SITE_1_PARTICIPANT}" "${SITE_1_BREV}"
copy_participant "${SITE_2_PARTICIPANT}" "${SITE_2_BREV}"

report_port_exposure

cat <<EOF

Next steps:
  Run the launch script inside each Brev Kubernetes environment:

    brev shell ${SERVER_BREV}
    IMAGE="${IMAGE}" bash ${REMOTE_DIR}/launch_brev_nvflare.sh ${SERVER_PARTICIPANT}

    brev shell ${SITE_1_BREV}
    IMAGE="${IMAGE}" SERVER_HOST="${SERVER_HOST}" bash ${REMOTE_DIR}/launch_brev_nvflare.sh ${SITE_1_PARTICIPANT}

    brev shell ${SITE_2_BREV}
    IMAGE="${IMAGE}" SERVER_HOST="${SERVER_HOST}" bash ${REMOTE_DIR}/launch_brev_nvflare.sh ${SITE_2_PARTICIPANT}

  Or run them non-interactively with brev exec after confirming kubectl and helm
  are configured in each Brev environment.
EOF
