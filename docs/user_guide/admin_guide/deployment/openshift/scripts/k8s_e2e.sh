#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the complete OpenShift/Kubernetes NVFlare e2e workflow by invoking the
three phase scripts in order:

  1. k8s_provision.sh
  2. k8s_deploy.sh
  3. k8s_submit_job.sh

Required environment:
  IMAGE  Parent container image pullable by the cluster. It must contain this
         NVFlare version with the K8S extra/Kubernetes Python client and the
         Python executable named by PARENT_PYTHON_PATH.

  JOB_IMAGE is required when IMAGE is parent-only, such as an image built from
         docker/Dockerfile.parent. ADMIN_IMAGE defaults to IMAGE and can use the
         parent image. JOB_IMAGE must contain NVFlare, Python, numpy, and the
         runtime tools needed by the job.

Common optional environment:
  KUBE_CMD=oc
  NAMESPACE=nvflare-e2e
  PROJECT_NAME=openshift_nvflare_e2e
  SERVER_NAME=nvflare-server
  SERVER_HOST=nvflare-server
  CLIENTS="site-1 site-2"
  ADMIN_USER=admin@nvidia.com
  WORK_DIR=/tmp/nvflare/openshift-e2e
  PARENT_PYTHON_PATH=python
  ADMIN_PYTHON_PATH=python
  STORAGE_CLASS=<cluster storage class>
  WORKSPACE_STORAGE=2Gi
  COPY_IMAGE=busybox:1.36  # must contain sh, sleep, and tar for oc cp
  PARENT_CPU=<optional parent pod CPU request, for example 500m>
  PARENT_MEMORY=<optional parent pod memory request, for example 1Gi>
  ADMIN_IMAGE=$IMAGE
  JOB_IMAGE=$IMAGE
  JOB_WAIT_TIMEOUT=900
  CLEAN_WORK_DIR=true

Examples:
  IMAGE=registry.example.com/nvflare-parent:dev \
  JOB_IMAGE=registry.example.com/nvflare-job:dev \
    bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_e2e.sh

  IMAGE=registry.example.com/nvflare-parent:dev \
  JOB_IMAGE=registry.example.com/nvflare-job:dev \
  PARENT_CPU=500m \
  PARENT_MEMORY=1Gi \
    bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_e2e.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLEAN_WORK_DIR="${CLEAN_WORK_DIR:-true}" bash "${SCRIPT_DIR}/k8s_provision.sh"
CLEAN_WORK_DIR=false bash "${SCRIPT_DIR}/k8s_deploy.sh"
CLEAN_WORK_DIR=false bash "${SCRIPT_DIR}/k8s_submit_job.sh"
