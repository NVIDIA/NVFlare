#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
CLUSTER_NAME="${CLUSTER_NAME:-gke-auto-test}"
LOCATION="${LOCATION:-us-central1}"
NETWORK_NAME="${NETWORK_NAME:-gke-test}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Set PROJECT_ID or configure a default gcloud project before deleting the cluster." >&2
  exit 1
fi

gcloud container clusters delete "${CLUSTER_NAME}" \
  --location "${LOCATION}" \
  --project "${PROJECT_ID}" \
  --quiet

# GKE can leave network-scoped firewall rules behind briefly after cluster deletion.
while IFS= read -r firewall_rule; do
  [[ -n "${firewall_rule}" ]] || continue
  gcloud compute firewall-rules delete "${firewall_rule}" \
    --project "${PROJECT_ID}" \
    --quiet
done < <(
  gcloud compute firewall-rules list \
    --filter="network=${NETWORK_NAME} AND name~'^gke-'" \
    --format="value(name)" \
    --project "${PROJECT_ID}"
)

if gcloud compute networks describe "${NETWORK_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute networks delete "${NETWORK_NAME}" \
    --project "${PROJECT_ID}" \
    --quiet
fi
