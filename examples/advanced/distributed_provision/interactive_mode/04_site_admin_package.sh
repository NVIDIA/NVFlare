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

nvflare package -e "${SERVER_ENDPOINT}" -p "${SITE_YAML}" --dir "${SITE_BUNDLE_DIR}"

