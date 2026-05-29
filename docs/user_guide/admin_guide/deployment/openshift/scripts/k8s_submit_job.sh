#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Export hello-numpy, submit it from an in-cluster admin pod, verify K8s launcher
job pods are created, and wait for the job to finish successfully.

Run these first:
  1. k8s_provision.sh
  2. k8s_deploy.sh

Required environment:
  IMAGE  Used as the default ADMIN_IMAGE and JOB_IMAGE. When used as
         ADMIN_IMAGE, it must contain the nvflare CLI, sh, sleep, and tar
         because the script uses oc cp to stage files into the admin pod.

Common optional environment:
  KUBE_CMD=oc
  NAMESPACE=nvflare-e2e
  PROJECT_NAME=openshift_nvflare_e2e
  CLIENTS="site-1 site-2"
  ADMIN_USER=admin@nvidia.com
  WORK_DIR=/tmp/nvflare/openshift-e2e
  ADMIN_IMAGE=$IMAGE  # must contain nvflare, sh, sleep, and tar for oc cp
  JOB_IMAGE=$IMAGE
  JOB_WAIT_TIMEOUT=900
  JOB_POD_APPEAR_TIMEOUT=180
  DELETE_ADMIN_POD_ON_EXIT=false
  DELETE_NAMESPACE_ON_EXIT=false

Example:
  IMAGE=registry.example.com/nvflare:dev \
    bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_submit_job.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k8s_common.sh
source "${SCRIPT_DIR}/k8s_common.sh"

init_k8s_env true
trap cleanup_on_exit EXIT
run_submit_job_phase
