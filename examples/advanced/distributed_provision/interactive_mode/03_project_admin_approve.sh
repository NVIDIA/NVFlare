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

echo >&2
echo "Share the rootca_fingerprint_sha256 value above with the site admin through a trusted out-of-band channel." >&2
