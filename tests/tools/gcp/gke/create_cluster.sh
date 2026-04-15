#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
CLUSTER_NAME="${CLUSTER_NAME:-gke-auto-test}"
LOCATION="${LOCATION:-us-central1}"
NETWORK_NAME="${NETWORK_NAME:-gke-test}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Set PROJECT_ID or configure a default gcloud project before creating the cluster." >&2
  exit 1
fi

if ! gcloud compute networks describe "${NETWORK_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute networks create "${NETWORK_NAME}" \
    --subnet-mode=auto \
    --description="nvflare-managed" \
    --project "${PROJECT_ID}"
fi

create_cmd=(
  gcloud
  container
  clusters
  create-auto
  "${CLUSTER_NAME}"
  --location
  "${LOCATION}"
  --project
  "${PROJECT_ID}"
  --network
  "${NETWORK_NAME}"
)

credentials_cmd=(
  gcloud
  container
  clusters
  get-credentials
  "${CLUSTER_NAME}"
  --location
  "${LOCATION}"
  --project
  "${PROJECT_ID}"
)

"${create_cmd[@]}"
"${credentials_cmd[@]}"

# Save kubeconfig for multicloud deploy scripts
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${REPO_ROOT}/.tmp/kubeconfigs"
KUBECONFIG="${REPO_ROOT}/.tmp/kubeconfigs/gcp.yaml" "${credentials_cmd[@]}"
echo "Kubeconfig saved to .tmp/kubeconfigs/gcp.yaml"

# Enable Filestore API for RWX PVCs
gcloud services enable file.googleapis.com --project "${PROJECT_ID}"

# Create Filestore StorageClass with correct VPC.
# The default standard-rwx assumes VPC "default" which fails on custom VPC clusters.
# Set FILESTORE_ZONE to pin to a specific zone (e.g. us-central1-a) if you hit
# zone capacity errors. Without it, the CSI driver picks the zone from the pod's node.
FILESTORE_ZONE="${FILESTORE_ZONE:-}"

if [[ -n "${FILESTORE_ZONE}" ]]; then
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: filestore-rwx
provisioner: filestore.csi.storage.gke.io
parameters:
  tier: BASIC_HDD
  network: ${NETWORK_NAME}
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
allowedTopologies:
  - matchLabelExpressions:
    - key: topology.gke.io/zone
      values:
      - ${FILESTORE_ZONE}
EOF
else
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: filestore-rwx
provisioner: filestore.csi.storage.gke.io
parameters:
  tier: BASIC_HDD
  network: ${NETWORK_NAME}
volumeBindingMode: Immediate
allowVolumeExpansion: true
EOF
fi
