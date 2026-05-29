#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Show an in-place live Rich pod table for OpenShift/Kubernetes pods created by
the NVFlare OpenShift scripts.

Usage:
  bash k8s_watch.sh [--once] [--interval SECONDS]

Required local Python package:
  rich

Common optional environment:
  KUBE_CMD=oc
  NAMESPACE=nvflare-e2e
  WORK_DIR=/tmp/nvflare/openshift-e2e

Examples:
  bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_watch.sh
  bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_watch.sh --once
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k8s_common.sh
source "${SCRIPT_DIR}/k8s_common.sh"

init_k8s_env false
require_cmd python3 "${KUBE_CMD}"

export KUBE_CMD NAMESPACE WORK_DIR LAST_JOB_ID_FILE
exec python3 "${SCRIPT_DIR}/k8s_watch.py" "$@"
