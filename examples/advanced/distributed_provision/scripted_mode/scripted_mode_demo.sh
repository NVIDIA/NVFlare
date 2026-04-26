#!/usr/bin/env bash
set -euo pipefail

# Scripted-mode demo: minimal request/approve/package flow with JSON stdout.
# Usage:
#   ./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
# Each site YAML is a single-site file with: name, org, type.

json_error() {
  local exit_code="$1"
  local error_code="$2"
  local message="$3"
  if command -v jq >/dev/null 2>&1; then
    jq -n -c \
      --argjson exit_code "${exit_code}" \
      --arg error_code "${error_code}" \
      --arg message "${message}" \
      '{status:"error", exit_code:$exit_code, error_code:$error_code, message:$message}'
  else
    printf '{"status":"error","exit_code":%s,"error_code":"%s","message":"%s"}\n' \
      "${exit_code}" "${error_code}" "${message}"
  fi
  exit "${exit_code}"
}

parse_site_yaml() {
  python3 - "$1" <<'PY'
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
}

PROJECT_NAME="${1:-}"
SERVER_ENDPOINT="${2:-}"
WORK_DIR="${3:-}"
if [[ $# -ge 3 ]]; then
  shift 3
fi
SITE_YAMLS=("$@")

if [[ -z "${PROJECT_NAME}" || -z "${SERVER_ENDPOINT}" || -z "${WORK_DIR}" || ${#SITE_YAMLS[@]} -eq 0 ]]; then
  json_error 2 "INVALID_ARGS" "Usage: $0 <project_name> <server_endpoint> <work_dir> <site_yaml...>"
fi

command -v nvflare >/dev/null 2>&1 || json_error 2 "MISSING_DEPENDENCY" "Missing nvflare"
command -v jq >/dev/null 2>&1 || json_error 2 "MISSING_DEPENDENCY" "Missing jq"
command -v openssl >/dev/null 2>&1 || json_error 2 "MISSING_DEPENDENCY" "Missing openssl"
command -v python3 >/dev/null 2>&1 || json_error 2 "MISSING_DEPENDENCY" "Missing python3"

CA_DIR="${WORK_DIR}/ca"
REQUEST_DIR="${WORK_DIR}/requests"
SIGNED_DIR="${WORK_DIR}/signed"
WORKSPACE_DIR="${WORK_DIR}/workspace"

mkdir -m 0700 -p "${CA_DIR}"
mkdir -m 0700 -p "${REQUEST_DIR}" "${SIGNED_DIR}" "${WORKSPACE_DIR}"

# 1) Root CA
jq -n -c '{step:"warning", scope:"cert init", message:"cert init --force regenerates the root CA and invalidates previously signed certs."}'
nvflare cert init --project "${PROJECT_NAME}" -o "${CA_DIR}" --force --format json >"${WORK_DIR}/ca.json"
ROOTCA_FP="$(openssl x509 -in "${CA_DIR}/rootCA.pem" -noout -fingerprint -sha256 | sed 's/^[Ss][Hh][Aa]256 Fingerprint=//')"

SITE_NAMES=()
REQUEST_DIRS=()
REQUEST_ZIPS=()
SIGNED_ZIPS=()
TMP_REQUEST_JSON=""
SEEN_SITE_NAMES=" "
trap 'rm -f "${TMP_REQUEST_JSON:-}"' EXIT

# 2) Request zip for each participant.
for site_yaml in "${SITE_YAMLS[@]}"; do
  SITE_INFO="$(parse_site_yaml "${site_yaml}")" || json_error 1 "INVALID_SITE_YAML" "Failed to parse ${site_yaml}"
  IFS=$'\t' read -r SITE_NAME SITE_ORG SITE_TYPE REQUEST_KIND REQUEST_ROLE <<<"${SITE_INFO}"

  if [[ "${SITE_NAME}" == "null" || -z "${SITE_NAME}" ]]; then
    json_error 1 "INVALID_SITE_YAML" "site yaml did not include a valid participant name: ${site_yaml}"
  fi
  if [[ "${SEEN_SITE_NAMES}" == *" ${SITE_NAME} "* ]]; then
    json_error 1 "DUPLICATE_SITE_NAME" "duplicate participant name in scripted mode input: ${SITE_NAME}"
  fi

  SITE_REQUEST_DIR="${REQUEST_DIR}/${SITE_NAME}"
  mkdir -m 0700 -p "${SITE_REQUEST_DIR}"
  TMP_REQUEST_JSON="$(mktemp "${WORK_DIR}/request_tmp.XXXXXX")"

  jq -n -c \
    --arg file "${site_yaml}" \
    --arg dir "${SITE_REQUEST_DIR}" \
    '{step:"warning", scope:"cert request", site_yaml:$file, message:"cert request --force regenerates local private keys, CSRs, request metadata, and request zips.", request_dir:$dir}'

  if [[ "${REQUEST_KIND}" == "user" ]]; then
    nvflare cert request user "${REQUEST_ROLE}" "${SITE_NAME}" \
      --org "${SITE_ORG}" \
      --project "${PROJECT_NAME}" \
      --out "${SITE_REQUEST_DIR}" \
      --force \
      --format json >"${TMP_REQUEST_JSON}"
  else
    nvflare cert request "${REQUEST_KIND}" "${SITE_NAME}" \
      --org "${SITE_ORG}" \
      --project "${PROJECT_NAME}" \
      --out "${SITE_REQUEST_DIR}" \
      --force \
      --format json >"${TMP_REQUEST_JSON}"
  fi

  REQUEST_ZIP="$(jq -r '.data.request_zip' <"${TMP_REQUEST_JSON}")"
  if [[ "${REQUEST_ZIP}" == "null" || -z "${REQUEST_ZIP}" ]]; then
    json_error 1 "INVALID_REQUEST_OUTPUT" "cert request output did not include a valid request_zip path."
  fi

  mv -f "${TMP_REQUEST_JSON}" "${WORK_DIR}/request_${SITE_NAME}.json"
  TMP_REQUEST_JSON=""
  SITE_NAMES+=("${SITE_NAME}")
  SEEN_SITE_NAMES+="${SITE_NAME} "
  REQUEST_DIRS+=("${SITE_REQUEST_DIR}")
  REQUEST_ZIPS+=("${REQUEST_ZIP}")

done

# 3) Approve each request zip.
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  REQUEST_ZIP="${REQUEST_ZIPS[$i]}"
  SIGNED_ZIP="${SIGNED_DIR}/${SITE_NAME}.signed.zip"
  nvflare cert approve "${REQUEST_ZIP}" -c "${CA_DIR}" --out "${SIGNED_ZIP}" --force --format json >"${WORK_DIR}/approve_${SITE_NAME}.json"
  SIGNED_ZIPS+=("${SIGNED_ZIP}")

done

# 4) Package each signed zip with the local request dir.
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  SITE_REQUEST_DIR="${REQUEST_DIRS[$i]}"
  SIGNED_ZIP="${SIGNED_ZIPS[$i]}"

  nvflare package "${SIGNED_ZIP}" \
    -e "${SERVER_ENDPOINT}" \
    --request-dir "${SITE_REQUEST_DIR}" \
    -w "${WORKSPACE_DIR}" \
    --force \
    --format json >"${WORK_DIR}/package_${SITE_NAME}.json"

done

# Summaries
jq -c '{step:"cert init", data:.data}' <"${WORK_DIR}/ca.json"
jq -n -c --arg fp "${ROOTCA_FP}" '{step:"verify rootca", fingerprint_sha256:$fp}'
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  jq -c --arg site "${SITE_NAME}" '{step:"cert request", site:$site, data:.data}' <"${WORK_DIR}/request_${SITE_NAME}.json"
  jq -c --arg site "${SITE_NAME}" '{step:"cert approve", site:$site, data:.data}' <"${WORK_DIR}/approve_${SITE_NAME}.json"
  jq -n -c --arg site "${SITE_NAME}" --arg fp "${ROOTCA_FP}" '{step:"verify rootca", site:$site, fingerprint_sha256:$fp}'
  jq -c --arg site "${SITE_NAME}" '{step:"package", site:$site, data:.data}' <"${WORK_DIR}/package_${SITE_NAME}.json"

done
