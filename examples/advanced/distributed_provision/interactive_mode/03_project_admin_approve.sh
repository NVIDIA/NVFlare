#!/usr/bin/env bash
set -euo pipefail

# Project Admin: approve a request zip and return a signed zip.
# Usage: ./03_project_admin_approve.sh <request_zip> <ca_dir> <signed_zip>

REQUEST_ZIP="${1:-}"
CA_DIR="${2:-}"
SIGNED_ZIP="${3:-}"

if [[ -z "${REQUEST_ZIP}" || -z "${CA_DIR}" || -z "${SIGNED_ZIP}" ]]; then
  echo "Usage: $0 <request_zip> <ca_dir> <signed_zip>" >&2
  exit 2
fi

mkdir -m 0700 -p "$(dirname "${SIGNED_ZIP}")"

nvflare cert approve "${REQUEST_ZIP}" -c "${CA_DIR}" --out "${SIGNED_ZIP}"

if command -v openssl >/dev/null 2>&1; then
  echo >&2
  echo "Share this rootCA.pem SHA256 fingerprint with the site admin for out-of-band verification:" >&2
  openssl x509 -in "${CA_DIR}/rootCA.pem" -noout -fingerprint -sha256 >&2
else
  echo "WARNING: openssl not found; share the rootCA.pem fingerprint with the site admin manually." >&2
fi
