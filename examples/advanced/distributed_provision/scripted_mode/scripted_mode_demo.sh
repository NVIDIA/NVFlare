#!/usr/bin/env bash
set -euo pipefail

# Scripted-mode demo: minimal, JSON stdout via --out-format json.
# Usage:
#   ./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
# Each site YAML is a single-site file with: name, org, type.

PROJECT_NAME="${1:-}"
SERVER_ENDPOINT="${2:-}"
WORK_DIR="${3:-}"
shift 3 || true
SITE_YAMLS=("$@")

if [[ -z "${PROJECT_NAME}" || -z "${SERVER_ENDPOINT}" || -z "${WORK_DIR}" || ${#SITE_YAMLS[@]} -eq 0 ]]; then
  echo "Usage: $0 <project_name> <server_endpoint> <work_dir> <site_yaml...>" >&2
  exit 2
fi

command -v nvflare >/dev/null 2>&1 || { echo "Missing nvflare" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "Missing jq" >&2; exit 2; }

CA_DIR="${WORK_DIR}/ca"
CSR_DIR="${WORK_DIR}/csr"
SIGNED_DIR="${WORK_DIR}/signed"
SITE_DIR="${WORK_DIR}/site"

mkdir -p "${CA_DIR}" "${CSR_DIR}" "${SIGNED_DIR}" "${SITE_DIR}"

# 1) Root CA (idempotent)
nvflare --out-format json cert init --project "${PROJECT_NAME}" -o "${CA_DIR}" --force >"${WORK_DIR}/ca.json"

SITE_NAMES=()
CSR_PATHS=()
KEY_PATHS=()

# 2) CSR for each site (uses --project-file)
for site_yaml in "${SITE_YAMLS[@]}"; do
  TMP_CSR_JSON="${WORK_DIR}/csr_tmp.json"
  nvflare --out-format json cert csr --project-file "${site_yaml}" -o "${CSR_DIR}" --force >"${TMP_CSR_JSON}"

  SITE_NAME="$(jq -r '.data.name' <"${TMP_CSR_JSON}")"
  CSR_PATH="$(jq -r '.data.csr' <"${TMP_CSR_JSON}")"
  KEY_PATH="$(jq -r '.data.key' <"${TMP_CSR_JSON}")"

  mv -f "${TMP_CSR_JSON}" "${WORK_DIR}/csr_${SITE_NAME}.json"
  SITE_NAMES+=("${SITE_NAME}")
  CSR_PATHS+=("${CSR_PATH}")
  KEY_PATHS+=("${KEY_PATH}")

done

# 3) Sign CSR for each site.
# This accepts the role proposed in each site-admin-generated CSR.
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  CSR_PATH="${CSR_PATHS[$i]}"
  SITE_SIGNED_DIR="${SIGNED_DIR}/${SITE_NAME}"
  mkdir -p "${SITE_SIGNED_DIR}"
  nvflare --out-format json cert sign -r "${CSR_PATH}" -c "${CA_DIR}" -o "${SITE_SIGNED_DIR}" --accept-csr-role --force >"${WORK_DIR}/sign_${SITE_NAME}.json"

done

# 4) Package for each site (uses site.yml)
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  KEY_PATH="${KEY_PATHS[$i]}"
  SITE_SIGNED_DIR="${SIGNED_DIR}/${SITE_NAME}"
  SITE_BUNDLE_DIR="${SITE_DIR}/${SITE_NAME}"
  mkdir -p "${SITE_BUNDLE_DIR}"

  cp -f "${KEY_PATH}" "${SITE_BUNDLE_DIR}/"
  cp -f "${SITE_SIGNED_DIR}/${SITE_NAME}.crt" "${SITE_BUNDLE_DIR}/" 2>/dev/null || true
  cp -f "${SITE_SIGNED_DIR}/${SITE_NAME}.pem" "${SITE_BUNDLE_DIR}/" 2>/dev/null || true
  if [[ ! -f "${SITE_BUNDLE_DIR}/${SITE_NAME}.crt" && ! -f "${SITE_BUNDLE_DIR}/${SITE_NAME}.pem" ]]; then
    echo "ERROR: no signed cert found for ${SITE_NAME} in ${SITE_SIGNED_DIR}" >&2
    exit 1
  fi
  cp -f "${CA_DIR}/rootCA.pem" "${SITE_BUNDLE_DIR}/"

  nvflare --out-format json package -e "${SERVER_ENDPOINT}" -p "${SITE_YAMLS[$i]}" --dir "${SITE_BUNDLE_DIR}" -w "${WORK_DIR}/workspace" >"${WORK_DIR}/package_${SITE_NAME}.json"

done

# Summaries
jq -r '{step:"cert init", data:.data} | @json' <"${WORK_DIR}/ca.json"
for i in "${!SITE_NAMES[@]}"; do
  SITE_NAME="${SITE_NAMES[$i]}"
  jq -r '{step:"cert csr", site:"'"${SITE_NAME}"'", data:.data} | @json' <"${WORK_DIR}/csr_${SITE_NAME}.json"
  jq -r '{step:"cert sign", site:"'"${SITE_NAME}"'", data:.data} | @json' <"${WORK_DIR}/sign_${SITE_NAME}.json"
  echo "{\"step\":\"package\",\"site\":\"${SITE_NAME}\",\"output\":\"${WORK_DIR}/package_${SITE_NAME}.json\"}"

done
