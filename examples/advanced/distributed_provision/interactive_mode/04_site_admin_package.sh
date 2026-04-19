#!/usr/bin/env bash
set -euo pipefail

# Site Admin: generate a startup kit from site.yml and a site bundle directory.
# The bundle directory must contain: <name>.key, <name>.crt (or .pem), rootCA.pem
# Usage: ./04_site_admin_package.sh <site_yaml> <server_endpoint> <site_bundle_dir>

SITE_YAML="${1:-}"
SERVER_ENDPOINT="${2:-}"
SITE_BUNDLE_DIR="${3:-}"

if [[ -z "${SITE_YAML}" || -z "${SERVER_ENDPOINT}" || -z "${SITE_BUNDLE_DIR}" ]]; then
  echo "Usage: $0 <site_yaml> <server_endpoint> <site_bundle_dir>" >&2
  exit 2
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl is required to verify the rootCA.pem fingerprint before packaging." >&2
  exit 2
fi

echo "Verify this rootCA.pem SHA256 fingerprint with the Project Admin through a trusted out-of-band channel before packaging:" >&2
openssl x509 -in "${SITE_BUNDLE_DIR}/rootCA.pem" -noout -fingerprint -sha256 >&2
read -r -p "Fingerprint verified out-of-band? [y/N] " CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
  echo "Aborting packaging until rootCA.pem fingerprint is verified." >&2
  exit 1
fi

nvflare package -e "${SERVER_ENDPOINT}" -p "${SITE_YAML}" --dir "${SITE_BUNDLE_DIR}"
