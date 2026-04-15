#!/usr/bin/env bash
set -euo pipefail

# Project Admin: initialize the root CA (run once per federation).
# Usage: ./01_project_admin_init_ca.sh <project_name> <ca_dir>

PROJECT_NAME="${1:-}"
CA_DIR="${2:-}"

if [[ -z "${PROJECT_NAME}" || -z "${CA_DIR}" ]]; then
  echo "Usage: $0 <project_name> <ca_dir>" >&2
  exit 2
fi

mkdir -p "${CA_DIR}"

# Root CA does not require org.
nvflare cert init --project "${PROJECT_NAME}" -o "${CA_DIR}"

