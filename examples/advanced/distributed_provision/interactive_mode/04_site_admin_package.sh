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

if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl is required to verify the rootCA.pem fingerprint before packaging." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to read rootCA.pem from ${SIGNED_ZIP}." >&2
  exit 2
fi

echo "Verify this rootCA.pem SHA256 fingerprint with the Project Admin through a trusted out-of-band channel before packaging:" >&2
python3 - "${SIGNED_ZIP}" <<'PY' | openssl x509 -noout -fingerprint -sha256 >&2
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "r") as zf:
    sys.stdout.buffer.write(zf.read("rootCA.pem"))
PY
read -r -p "Fingerprint verified out-of-band? [y/N] " CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
  echo "Aborting packaging until rootCA.pem fingerprint is verified." >&2
  exit 1
fi

PACKAGE_ARGS=(nvflare package "${SIGNED_ZIP}" -e "${SERVER_ENDPOINT}" --request-dir "${REQUEST_DIR}")
if [[ -n "${PROJECT_FILE}" ]]; then
  PACKAGE_ARGS+=(--project-file "${PROJECT_FILE}")
fi

"${PACKAGE_ARGS[@]}"
