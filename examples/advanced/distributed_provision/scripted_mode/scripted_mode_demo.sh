#!/usr/bin/env bash
set -euo pipefail

# Scripted-mode demo for automating distributed provisioning.
#
# This file intentionally uses only the public CLI commands that an automation
# service should call:
#   1. nvflare cert init
#   2. nvflare cert request
#   3. nvflare cert approve
#   4. nvflare package
#
# The demo runs all roles on one machine so it can be executed end-to-end. In a
# real deployment, "cert request" and "package" usually run on the requester
# side, while "cert init" and "cert approve" run on the Project Admin side.
#
# Artifacts written under <work_dir>:
#   ca/                         Project Admin CA material
#   requests/<name>/            Requester private key, CSR, metadata, request zip
#   signed/<name>.signed.zip    Project Admin signed response zip
#   workspace/                  Generated startup kits
#   request_<name>.json         JSON output from cert request
#   approve_<name>.json         JSON output from cert approve
#   package_<name>.json         JSON output from package
#
# Usage:
#   ./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
# Each site YAML is a single-participant identity file with: name, org, type.

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

require_command() {
  command -v "$1" >/dev/null 2>&1 || json_error 2 "MISSING_DEPENDENCY" "Missing $1"
}

json_event() {
  local step="$1"
  local message="$2"
  jq -n -c --arg step "${step}" --arg message "${message}" '{step:$step, message:$message}'
}

parse_site_yaml() {
  # Translate the small single-participant YAML used by this example into the
  # public cert request command shape. This keeps the example input simple while
  # still covering sites, servers, and admin users.
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

run_cert_request() {
  local site_yaml="$1"
  local site_info site_name site_org site_type request_kind request_role
  local site_request_dir request_json request_zip

  site_info="$(parse_site_yaml "${site_yaml}")" || json_error 1 "INVALID_SITE_YAML" "Failed to parse ${site_yaml}"
  IFS=$'\t' read -r site_name site_org site_type request_kind request_role <<<"${site_info}"

  if [[ "${site_name}" == "null" || -z "${site_name}" ]]; then
    json_error 1 "INVALID_SITE_YAML" "site yaml did not include a valid participant name: ${site_yaml}"
  fi
  if [[ "${SEEN_SITE_NAMES}" == *" ${site_name} "* ]]; then
    json_error 1 "DUPLICATE_SITE_NAME" "duplicate participant name in scripted mode input: ${site_name}"
  fi

  site_request_dir="${REQUESTS_DIR}/${site_name}"
  request_json="${WORK_DIR}/request_${site_name}.json"
  mkdir -m 0700 -p "${site_request_dir}"

  jq -n -c \
    --arg site "${site_name}" \
    --arg file "${site_yaml}" \
    --arg dir "${site_request_dir}" \
    '{
      step:"cert request",
      site:$site,
      site_yaml:$file,
      request_dir:$dir,
      note:"Requester creates a private key locally and sends only the request zip to the Project Admin."
    }'

  if [[ "${request_kind}" == "user" ]]; then
    nvflare cert request user "${request_role}" "${site_name}" \
      --org "${site_org}" \
      --project "${PROJECT_NAME}" \
      --out "${site_request_dir}" \
      --force \
      --format json >"${request_json}"
  else
    nvflare cert request "${request_kind}" "${site_name}" \
      --org "${site_org}" \
      --project "${PROJECT_NAME}" \
      --out "${site_request_dir}" \
      --force \
      --format json >"${request_json}"
  fi

  request_zip="$(jq -r '.data.request_zip' <"${request_json}")"
  if [[ "${request_zip}" == "null" || -z "${request_zip}" ]]; then
    json_error 1 "INVALID_REQUEST_OUTPUT" "cert request output did not include a valid request_zip path."
  fi

  SITE_NAMES+=("${site_name}")
  REQUEST_DIRS+=("${site_request_dir}")
  REQUEST_ZIPS+=("${request_zip}")
  SEEN_SITE_NAMES+="${site_name} "
}

run_cert_approve() {
  local index="$1"
  local site_name request_zip signed_zip approve_json rootca_fp

  site_name="${SITE_NAMES[$index]}"
  request_zip="${REQUEST_ZIPS[$index]}"
  signed_zip="${SIGNED_DIR}/${site_name}.signed.zip"
  approve_json="${WORK_DIR}/approve_${site_name}.json"

  jq -n -c \
    --arg site "${site_name}" \
    --arg request_zip "${request_zip}" \
    --arg signed_zip "${signed_zip}" \
    '{
      step:"cert approve",
      site:$site,
      request_zip:$request_zip,
      signed_zip:$signed_zip,
      note:"Project Admin validates the request zip, signs the CSR, and returns a signed zip."
    }'

  nvflare cert approve "${request_zip}" \
    -c "${CA_DIR}" \
    --out "${signed_zip}" \
    --force \
    --format json >"${approve_json}"

  rootca_fp="$(jq -r '.data.rootca_fingerprint_sha256' <"${approve_json}")"
  if [[ "${rootca_fp}" == "null" || -z "${rootca_fp}" ]]; then
    json_error 1 "INVALID_APPROVE_OUTPUT" "cert approve output did not include rootca_fingerprint_sha256."
  fi

  SIGNED_ZIPS+=("${signed_zip}")
  ROOTCA_FPS+=("${rootca_fp}")
}

run_package() {
  local index="$1"
  local site_name site_request_dir signed_zip package_json

  site_name="${SITE_NAMES[$index]}"
  site_request_dir="${REQUEST_DIRS[$index]}"
  signed_zip="${SIGNED_ZIPS[$index]}"
  package_json="${WORK_DIR}/package_${site_name}.json"

  jq -n -c \
    --arg site "${site_name}" \
    --arg signed_zip "${signed_zip}" \
    --arg request_dir "${site_request_dir}" \
    '{
      step:"package",
      site:$site,
      signed_zip:$signed_zip,
      request_dir:$request_dir,
      note:"Requester combines signed zip, local private key, endpoint, and expected root CA fingerprint."
    }'

  nvflare package "${signed_zip}" \
    -e "${SERVER_ENDPOINT}" \
    --request-dir "${site_request_dir}" \
    --expected-rootca-fingerprint "${ROOTCA_FPS[$index]}" \
    -w "${WORKSPACE_DIR}" \
    --force \
    --format json >"${package_json}"
}

emit_summary() {
  # Emit compact newline-delimited JSON so automation can pipe the script output
  # into jq, a log collector, or a CI system.
  jq -c '{step:"cert init result", data:.data}' <"${WORK_DIR}/ca.json"
  for i in "${!SITE_NAMES[@]}"; do
    local site_name="${SITE_NAMES[$i]}"
    jq -c --arg site "${site_name}" '{step:"cert request result", site:$site, data:.data}' \
      <"${WORK_DIR}/request_${site_name}.json"
    jq -c --arg site "${site_name}" '{step:"cert approve result", site:$site, data:.data}' \
      <"${WORK_DIR}/approve_${site_name}.json"
    jq -n -c --arg site "${site_name}" --arg fp "${ROOTCA_FPS[$i]}" \
      '{step:"rootca verification input", site:$site, expected_rootca_fingerprint_sha256:$fp}'
    jq -c --arg site "${site_name}" '{step:"package result", site:$site, data:.data}' \
      <"${WORK_DIR}/package_${site_name}.json"
  done
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

require_command nvflare
require_command jq
require_command python3

CA_DIR="${WORK_DIR}/ca"
REQUESTS_DIR="${WORK_DIR}/requests"
SIGNED_DIR="${WORK_DIR}/signed"
WORKSPACE_DIR="${WORK_DIR}/workspace"

mkdir -m 0700 -p "${CA_DIR}"
mkdir -m 0700 -p "${REQUESTS_DIR}" "${SIGNED_DIR}" "${WORKSPACE_DIR}"

SITE_NAMES=()
REQUEST_DIRS=()
REQUEST_ZIPS=()
SIGNED_ZIPS=()
ROOTCA_FPS=()
SEEN_SITE_NAMES=" "

# Phase 1: Project Admin creates or refreshes the project CA.
json_event "cert init" "Project Admin initializes the project CA. --force replaces existing CA material in this demo work directory."
nvflare cert init --project "${PROJECT_NAME}" -o "${CA_DIR}" --force --format json >"${WORK_DIR}/ca.json"

# Phase 2: each requester creates a request zip. The private key remains under
# requests/<name>/ and is never copied into the request zip.
for site_yaml in "${SITE_YAMLS[@]}"; do
  run_cert_request "${site_yaml}"
done

# Phase 3: Project Admin approves each request zip and captures the root CA
# fingerprint that requesters should verify out-of-band.
for i in "${!SITE_NAMES[@]}"; do
  run_cert_approve "${i}"
done

# Phase 4: each requester packages its own startup kit. Automation should use
# --expected-rootca-fingerprint; interactive humans can use --confirm-rootca.
for i in "${!SITE_NAMES[@]}"; do
  run_package "${i}"
done

emit_summary
