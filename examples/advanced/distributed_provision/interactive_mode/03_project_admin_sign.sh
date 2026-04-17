#!/usr/bin/env bash
set -euo pipefail

# Project Admin: sign a CSR and return signed cert + rootCA.pem.
# Usage:
#   ./03_project_admin_sign.sh <csr_path> <ca_dir> <out_dir>
#     Accept the role proposed by the trusted site-admin-generated CSR.
#   ./03_project_admin_sign.sh <csr_path> <ca_dir> <out_dir> <type>
#     Override the role while signing.
# Usage: ./03_project_admin_sign.sh <csr_path> <ca_dir> <out_dir> [type]

CSR_PATH="${1:-}"
CA_DIR="${2:-}"
OUT_DIR="${3:-}"
OVERRIDE_TYPE="${4:-}"

if [[ -z "${CSR_PATH}" || -z "${CA_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: $0 <csr_path> <ca_dir> <out_dir> [type]" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

if [[ -n "${OVERRIDE_TYPE}" ]]; then
  nvflare cert sign -r "${CSR_PATH}" -t "${OVERRIDE_TYPE}" -c "${CA_DIR}" -o "${OUT_DIR}"
else
  nvflare cert sign -r "${CSR_PATH}" -c "${CA_DIR}" -o "${OUT_DIR}" --accept-csr-role
fi
