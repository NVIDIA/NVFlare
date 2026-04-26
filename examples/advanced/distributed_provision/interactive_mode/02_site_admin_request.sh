#!/usr/bin/env bash
set -euo pipefail

# Site Admin: create a distributed provisioning request zip from site.yml.
# Usage: ./02_site_admin_request.sh <project_name> <site_yaml> <request_dir>

PROJECT_NAME="${1:-}"
SITE_YAML="${2:-}"
REQUEST_DIR="${3:-}"

if [[ -z "${PROJECT_NAME}" || -z "${SITE_YAML}" || -z "${REQUEST_DIR}" ]]; then
  echo "Usage: $0 <project_name> <site_yaml> <request_dir>" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to read ${SITE_YAML}." >&2
  exit 2
fi

SITE_INFO="$(
  python3 - "${SITE_YAML}" <<'PY'
import sys

import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if not isinstance(data, dict):
    raise SystemExit(f"{path}: site yaml must be a mapping")

name = data.get("name")
org = data.get("org")
site_type = data.get("type")
missing = [field for field, value in (("name", name), ("org", org), ("type", site_type)) if not value]
if missing:
    raise SystemExit(f"{path}: missing required field(s): {', '.join(missing)}")

role = ""
if site_type == "client":
    kind = "site"
elif site_type == "server":
    kind = "server"
elif site_type == "org_admin":
    kind = "user"
    role = "org-admin"
elif site_type in ("lead", "member"):
    kind = "user"
    role = site_type
else:
    raise SystemExit(f"{path}: unsupported type {site_type!r}")

print(f"{name}\t{org}\t{site_type}\t{kind}\t{role}")
PY
)"

IFS=$'\t' read -r SITE_NAME SITE_ORG SITE_TYPE REQUEST_KIND REQUEST_ROLE <<<"${SITE_INFO}"

mkdir -m 0700 -p "${REQUEST_DIR}"

if [[ "${REQUEST_KIND}" == "user" ]]; then
  nvflare cert request user "${REQUEST_ROLE}" "${SITE_NAME}" \
    --org "${SITE_ORG}" \
    --project "${PROJECT_NAME}" \
    --out "${REQUEST_DIR}"
else
  nvflare cert request "${REQUEST_KIND}" "${SITE_NAME}" \
    --org "${SITE_ORG}" \
    --project "${PROJECT_NAME}" \
    --out "${REQUEST_DIR}"
fi
