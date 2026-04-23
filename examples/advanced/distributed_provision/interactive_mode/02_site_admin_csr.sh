#!/usr/bin/env bash
set -euo pipefail

# Site Admin: generate a CSR and private key from site.yml.
# Usage: ./02_site_admin_csr.sh <site_yaml> <csr_dir>

SITE_YAML="${1:-}"
CSR_DIR="${2:-}"

if [[ -z "${SITE_YAML}" || -z "${CSR_DIR}" ]]; then
  echo "Usage: $0 <site_yaml> <csr_dir>" >&2
  exit 2
fi

mkdir -m 0700 -p "${CSR_DIR}"

nvflare cert csr --project-file "${SITE_YAML}" -o "${CSR_DIR}"
