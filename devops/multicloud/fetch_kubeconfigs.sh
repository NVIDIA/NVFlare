#!/usr/bin/env bash
# Populate .tmp/kubeconfigs/{aws,gcp}.yaml for existing clusters.
# Use when clusters are already created (skip create_cluster.sh) and you
# just need kubeconfigs to drive deploy.py / k8sview.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/.tmp/kubeconfigs"
mkdir -p "${OUT_DIR}"

# ----- AWS / EKS -----
AWS_CLUSTER_YAML="${REPO_ROOT}/devops/aws/eks/cluster.yaml"
AWS_CLUSTER_NAME="$(grep 'name:' "${AWS_CLUSTER_YAML}" | head -1 | awk '{print $2}')"
AWS_REGION="$(grep 'region:' "${AWS_CLUSTER_YAML}" | awk '{print $2}')"

aws eks update-kubeconfig \
  --name "${AWS_CLUSTER_NAME}" \
  --region "${AWS_REGION}" \
  --kubeconfig "${OUT_DIR}/aws.yaml"
echo "Wrote ${OUT_DIR}/aws.yaml (cluster=${AWS_CLUSTER_NAME} region=${AWS_REGION})"

# ----- GCP / GKE -----
GCP_PROJECT="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
GCP_CLUSTER="${CLUSTER_NAME:-gke-auto-test}"
GCP_LOCATION="${LOCATION:-us-central1}"

if [[ -z "${GCP_PROJECT}" ]]; then
  echo "PROJECT_ID not set and no gcloud default project" >&2
  exit 1
fi

KUBECONFIG="${OUT_DIR}/gcp.yaml" gcloud container clusters get-credentials \
  "${GCP_CLUSTER}" \
  --location "${GCP_LOCATION}" \
  --project "${GCP_PROJECT}"
echo "Wrote ${OUT_DIR}/gcp.yaml (cluster=${GCP_CLUSTER} location=${GCP_LOCATION} project=${GCP_PROJECT})"
