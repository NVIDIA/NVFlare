#!/usr/bin/env bash
set -euo pipefail

# Site Admin: generate a startup kit from the signed zip and local request dir.
# The request dir must be the folder created by 'nvflare cert request'.
# Usage: ./04_site_admin_package.sh <signed_zip> <server_endpoint> <request_dir> [project_file]

SIGNED_ZIP="${1:-}"
SERVER_ENDPOINT="${2:-}"
REQUEST_DIR="${3:-}"
PROJECT_FILE="${4:-}"

if [[ -z "${SIGNED_ZIP}" || -z "${SERVER_ENDPOINT}" || -z "${REQUEST_DIR}" ]]; then
  echo "Usage: $0 <signed_zip> <server_endpoint> <request_dir> [project_file]" >&2
  exit 2
fi

PACKAGE_ARGS=(nvflare package "${SIGNED_ZIP}" -e "${SERVER_ENDPOINT}" --request-dir "${REQUEST_DIR}" --confirm-rootca)
if [[ -n "${PROJECT_FILE}" ]]; then
  PACKAGE_ARGS+=(--project-file "${PROJECT_FILE}")
fi

"${PACKAGE_ARGS[@]}"
