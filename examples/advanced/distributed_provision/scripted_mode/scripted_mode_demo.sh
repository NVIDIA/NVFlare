#!/usr/bin/env bash
set -euo pipefail

# Scripted-mode demo: minimal, JSON stdout via --out-format json.
# Usage:
#   ./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
# Each site YAML is a single-site file with: name, org, type.

json_error() {
  local exit_code="$1"
  local error_code="$2"
  local message="$3"
  jq -n -r \
    --argjson exit_code "${exit_code}" \
    --arg error_code "${error_code}" \
    --arg message "${message}" \
    '{status:"error", exit_code:$exit_code, error_code:$error_code, message:$message}'
  exit "${exit_code}"
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

CA_DIR="${WORK_DIR}/ca"
CSR_DIR="${WORK_DIR}/csr"
SIGNED_DIR="${WORK_DIR}/signed"
SITE_DIR="${WORK_DIR}/site"

mkdir -m 0700 -p "${CA_DIR}"
mkdir -m 0700 -p "${CSR_DIR}" "${SIGNED_DIR}" "${SITE_DIR}"

# 1) Root CA
jq -n -r '{step:"warning", scope:"cert init", message:"cert init --force regenerates the root CA and invalidates previously signed certs."} | @json'
nvflare --out-format json cert init --project "${PROJECT_NAME}" -o "${CA_DIR}" --force >"${WORK_DIR}/ca.json"
ROOTCA_FP="$(openssl x509 -in "${CA_DIR}/rootCA.pem" -noout -fingerprint -sha256 | sed 's/^[Ss][Hh][Aa]256 Fingerprint=//')"

SITE_NAMES=()
CSR_PATHS=()
KEY_PATHS=()
TMP_CSR_JSON=""
trap 'rm -f "${TMP_CSR_JSON:-}"' EXIT

# 2) CSR for each site (uses --project-file)
for site_yaml in "${SITE_YAMLS[@]}"; do
  TMP_CSR_JSON="$(mktemp "${WORK_DIR}/csr_tmp.XXXXXX.json")"
  jq -n -r --arg file "${site_yaml}" --arg dir "${CSR_DIR}" '{step:"warning", scope:"cert csr", site_yaml:$file, message:"cert csr --force regenerates participant private keys and CSRs for existing names.", csr_dir:$dir} | @json'
  nvflare --out-format json cert csr --project-file "${site_yaml}" -o "${CSR_DIR}" --force >"${TMP_CSR_JSON}"

  SITE_NAME="$(jq -r '.data.name' <"${TMP_CSR_JSON}")"
  CSR_PATH="$(jq -r '.data.csr' <"${TMP_CSR_JSON}")"
  KEY_PATH="$(jq -r '.data.key' <"${TMP_CSR_JSON}")"

  if [[ "${SITE_NAME}" == "null" || -z "${SITE_NAME}" ]]; then
    json_error 1 "INVALID_CSR_OUTPUT" "cert csr output did not include a valid participant name."
  fi
  for existing_site in "${SITE_NAMES[@]}"; do
    if [[ "${existing_site}" == "${SITE_NAME}" ]]; then
      json_error 1 "DUPLICATE_SITE_NAME" "duplicate participant name in scripted mode input: ${SITE_NAME}"
    fi
  done

  mv -f "${TMP_CSR_JSON}" "${WORK_DIR}/csr_${SITE_NAME}.json"
  TMP_CSR_JSON=""
  SITE_NAMES+=("${SITE_NAME}")
  CSR_PATHS+=("${CSR_PATH}")
  KEY_PATHS+=("${KEY_PATH}")

done

# 3) Sign CSR for each site.
# This demo accepts the role proposed in each site-admin-generated CSR.
# In production, review the proposed role for each CSR and use -t to override it
# if the Project Admin does not want to trust the embedded request.
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  CSR_PATH="${CSR_PATHS[$i]}"
  SITE_SIGNED_DIR="${SIGNED_DIR}/${SITE_NAME}"
  mkdir -m 0700 -p "${SITE_SIGNED_DIR}"
  nvflare --out-format json cert sign -r "${CSR_PATH}" -c "${CA_DIR}" -o "${SITE_SIGNED_DIR}" --accept-csr-role --force >"${WORK_DIR}/sign_${SITE_NAME}.json"

done

# 4) Package for each site (uses site.yml)
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  KEY_PATH="${KEY_PATHS[$i]}"
  SITE_SIGNED_DIR="${SIGNED_DIR}/${SITE_NAME}"
  SITE_BUNDLE_DIR="${SITE_DIR}/${SITE_NAME}"
  mkdir -m 0700 -p "${SITE_BUNDLE_DIR}"

  install -m 0600 "${KEY_PATH}" "${SITE_BUNDLE_DIR}/"
  if [[ ! -f "${SITE_SIGNED_DIR}/${SITE_NAME}.crt" ]]; then
    json_error 1 "SIGNED_CERT_NOT_FOUND" "signed cert not found for ${SITE_NAME}: ${SITE_SIGNED_DIR}/${SITE_NAME}.crt"
  fi
  cp -f "${SITE_SIGNED_DIR}/${SITE_NAME}.crt" "${SITE_BUNDLE_DIR}/"
  cp -f "${CA_DIR}/rootCA.pem" "${SITE_BUNDLE_DIR}/"

  nvflare --out-format json package -e "${SERVER_ENDPOINT}" -p "${SITE_YAMLS[$i]}" --dir "${SITE_BUNDLE_DIR}" -w "${WORK_DIR}/workspace" >"${WORK_DIR}/package_${SITE_NAME}.json"

done

# Summaries
jq -r '{step:"cert init", data:.data} | @json' <"${WORK_DIR}/ca.json"
jq -n -r --arg fp "${ROOTCA_FP}" '{step:"verify rootca", fingerprint_sha256:$fp} | @json'
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  jq -r --arg site "${SITE_NAME}" '{step:"cert csr", site:$site, data:.data} | @json' <"${WORK_DIR}/csr_${SITE_NAME}.json"
  jq -r --arg site "${SITE_NAME}" '{step:"cert sign", site:$site, data:.data} | @json' <"${WORK_DIR}/sign_${SITE_NAME}.json"
  jq -n -r --arg site "${SITE_NAME}" --arg fp "${ROOTCA_FP}" '{step:"verify rootca", site:$site, fingerprint_sha256:$fp} | @json'
  jq -n -r --arg site "${SITE_NAME}" --arg out "${WORK_DIR}/package_${SITE_NAME}.json" '{step:"package", site:$site, output:$out} | @json'

done
