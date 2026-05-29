#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare NVFlare K8s startup kits, deploy them into OpenShift/Kubernetes, and
verify the parent server/client pods are running.

Run k8s_provision.sh before this script.

Required environment:
  IMAGE  Parent container image pullable by the cluster. It must contain this
         NVFlare version with the K8S extra/Kubernetes Python client and the
         Python executable named by PARENT_PYTHON_PATH.

Common optional environment:
  KUBE_CMD=oc
  NAMESPACE=nvflare-e2e
  PROJECT_NAME=openshift_nvflare_e2e
  SERVER_NAME=nvflare-server
  CLIENTS="site-1 site-2"
  WORK_DIR=/tmp/nvflare/openshift-e2e
  PARENT_PYTHON_PATH=python
  STORAGE_CLASS=<cluster storage class>
  WORKSPACE_STORAGE=2Gi
  COPY_IMAGE=busybox:1.36  # must contain sh, sleep, and tar for oc cp
  PARENT_CPU=<optional parent pod CPU request, for example 500m>
  PARENT_MEMORY=<optional parent pod memory request, for example 1Gi>
  PARENT_IMAGE_PULL_SECRETS="registry-secret another-secret"
  JOB_IMAGE_PULL_SECRETS="registry-secret another-secret"

Example:
  IMAGE=registry.example.com/nvflare-parent:dev \
    bash docs/user_guide/admin_guide/deployment/openshift/scripts/k8s_deploy.sh
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
run_deploy_phase
