#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate NVFlare OpenShift e2e startup kits for:
  - server: nvflare-server
  - clients: site-1 site-2
  - admin: admin@nvidia.com

This script writes project.yml and runs:
  nvflare provision -p project.yml -w <workspace> --force

Output defaults to:
  /tmp/nvflare/openshift-e2e/workspace/<PROJECT_NAME>/prod_00

Common optional environment:
  PROJECT_NAME=openshift_nvflare_e2e
  SERVER_NAME=nvflare-server
  SERVER_HOST=nvflare-server
  CLIENTS="site-1 site-2"
  ADMIN_USER=admin@nvidia.com
  ADMIN_ROLE=lead
  ORG=nvidia
  WORK_DIR=/tmp/nvflare/openshift-e2e
  CLEAN_WORK_DIR=true

Example:
  bash examples/devops/openshift/scripts/k8s_provision.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k8s_common.sh
source "${SCRIPT_DIR}/k8s_common.sh"

CLEAN_WORK_DIR="${CLEAN_WORK_DIR:-true}"
init_k8s_env false
run_provision_phase
