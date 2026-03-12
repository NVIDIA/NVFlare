#!/usr/bin/env bash

set -euo pipefail

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
  --overwrite-existing
