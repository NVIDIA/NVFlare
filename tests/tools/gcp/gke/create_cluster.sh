#!/usr/bin/env bash

set -euo pipefail

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
