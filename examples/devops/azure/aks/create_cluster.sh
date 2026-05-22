#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KUBECONFIG="${REPO_ROOT}/.tmp/kubeconfigs/azure.yaml"
export KUBECONFIG
mkdir -p "$(dirname -- "${KUBECONFIG}")"

RESOURCE_GROUP="${RESOURCE_GROUP:-myResourceGroup}"
CLUSTER_NAME="${CLUSTER_NAME:-myAKSAutomaticCluster}"
LOCATION="${LOCATION:-westus2}"

az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}"
az aks create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CLUSTER_NAME}" \
  --location "${LOCATION}" \
  --sku automatic

az aks get-credentials \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CLUSTER_NAME}" \
  --file "${KUBECONFIG}" \
  --overwrite-existing
echo "Kubeconfig saved to ${KUBECONFIG}"
echo "Using AKS default StorageClass managed-csi for RWO PVCs"
