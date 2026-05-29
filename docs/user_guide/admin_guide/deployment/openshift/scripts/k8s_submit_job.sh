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
  ADMIN_IMAGE or IMAGE  Image for the temporary admin pod. It must contain the
         NVFlare package and the Python executable named by ADMIN_PYTHON_PATH.
         The parent IMAGE can be used as ADMIN_IMAGE.
  JOB_IMAGE or IMAGE  Image for dynamically created job pods. It must contain
         NVFlare, Python, numpy, and the runtime tools needed by the job.

Common optional environment:
  KUBE_CMD=oc
  NAMESPACE=nvflare-e2e
  PROJECT_NAME=openshift_nvflare_e2e
  CLIENTS="site-1 site-2"
  ADMIN_USER=admin@nvidia.com
  WORK_DIR=/tmp/nvflare/openshift-e2e
  ADMIN_IMAGE=$IMAGE
  ADMIN_PYTHON_PATH=python
  JOB_IMAGE=$IMAGE
  JOB_WAIT_TIMEOUT=900
  JOB_POD_APPEAR_TIMEOUT=180
  DELETE_ADMIN_POD_ON_EXIT=false
  DELETE_NAMESPACE_ON_EXIT=false

Example:
  IMAGE=registry.example.com/nvflare-parent:dev \
  JOB_IMAGE=registry.example.com/nvflare-job:dev \
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

init_k8s_env false
trap cleanup_on_exit EXIT
run_submit_job_phase
