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

if gcloud compute networks describe "${NETWORK_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute networks delete "${NETWORK_NAME}" \
    --project "${PROJECT_ID}" \
    --quiet
fi
