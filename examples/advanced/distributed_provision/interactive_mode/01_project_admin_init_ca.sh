#!/usr/bin/env bash
set -euo pipefail

# Project Admin: initialize the root CA (run once per federation).
# Usage: ./01_project_admin_init_ca.sh <project_name> <project_org> <ca_dir>

PROJECT_NAME="${1:-}"
PROJECT_ORG="${2:-}"
CA_DIR="${3:-}"

if [[ -z "${PROJECT_NAME}" || -z "${PROJECT_ORG}" || -z "${CA_DIR}" ]]; then
  echo "Usage: $0 <project_name> <project_org> <ca_dir>" >&2
  exit 2
fi

mkdir -m 0700 -p "${CA_DIR}"

nvflare cert init --project "${PROJECT_NAME}" --org "${PROJECT_ORG}" -o "${CA_DIR}"
