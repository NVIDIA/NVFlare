#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RESOURCE_GROUP="${RESOURCE_GROUP:-myResourceGroup}"
CLUSTER_NAME="${CLUSTER_NAME:-myAKSAutomaticCluster}"
LOCATION="${LOCATION:-westus2}"

az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}"
az aks create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CLUSTER_NAME}" \
  --location "${LOCATION}" \
  --sku automatic

# Merge credentials into default kubeconfig
az aks get-credentials \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CLUSTER_NAME}" \
  --overwrite-existing

# Save kubeconfig for multicloud deploy scripts
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${REPO_ROOT}/.tmp/kubeconfigs"
az aks get-credentials \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CLUSTER_NAME}" \
  --file "${REPO_ROOT}/.tmp/kubeconfigs/azure.yaml" \
  --overwrite-existing
echo "Kubeconfig saved to .tmp/kubeconfigs/azure.yaml"
echo "Using AKS default StorageClass managed-csi for RWO PVCs"
