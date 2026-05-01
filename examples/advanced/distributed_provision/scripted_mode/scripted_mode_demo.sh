#!/usr/bin/env bash
set -euo pipefail

# Simple scripted-mode demo for the same federation used by interactive_mode.
#
# Initial setup:
#   1. Project Admin initializes the CA with provision version 00.
#
# What the default mode automates:
#   1. Requesters create request zips from the checked-in participant YAML files.
#   2. The local demo copies request zips to simulate handoff to Project Admin.
#   3. Project Admin approves each request zip, adding the server endpoint from
#      project_profile.yaml to the signed zip.
#   4. Requesters package startup kits from the signed zip and local request
#      material, with an expected root CA fingerprint for verification.
#
# What --add mode automates:
#   1. Add one new participant after the project has started.
#   2. Reuse the existing CA and skip cert init.
#   3. Run request, approve, and package for only that new participant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="fed_project"
PROVISION_VERSION="00"

# Demo inputs. These YAML files are checked in the parent example directory.
PARTICIPANTS=(
  "server.example.com:server.yaml"
  "site-1:site-1.yaml"
  "site-2:site-2.yaml"
  "alice@nvidia.com:alice.yaml"
)

# Helper functions used by the setup and automation steps.
usage() {
  cat <<EOF
Usage:
  $0 [work_dir]
  $0 --add <name> <participant_yaml> [work_dir]

Runs the scripted distributed provisioning demo with the same inputs as
the parent distributed_provision directory.

Run from:
  examples/advanced/distributed_provision

Example:
  ./scripted_mode/scripted_mode_demo.sh

Default work_dir: ./distprov_demo, relative to your current directory.
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

json_field() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path = sys.argv[1]
field_path = sys.argv[2].split(".")

with open(path, "r", encoding="utf-8") as f:
    value = json.load(f)

for field in field_path:
    value = value[field]

print(value)
PY
}

show() {
  printf '\n== %s ==\n' "$1"
}

run_logged() {
  echo "+ $*"
  "$@"
}

run_logged_to_file() {
  local output_file="$1"
  shift
  echo "+ $* > ${output_file}"
  "$@" >"${output_file}"
}

yaml_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${BASE_DIR}" "$1" ;;
  esac
}

request_participant() {
  local name="$1"
  local yaml="$2"
  local output_file="$3"

  run_logged_to_file "${output_file}" \
    nvflare cert request --participant "$(yaml_path "${yaml}")" --out "${name}" --force --format json
}

approve_participant() {
  local name="$1"
  local output_file="$2"

  run_logged_to_file "${output_file}" \
    nvflare cert approve "${name}.request.zip" --ca-dir ca --profile "${BASE_DIR}/project_profile.yaml" --force --format json
}

package_participant() {
  local name="$1"
  local approve_json="$2"
  local output_file="$3"
  local rootca_fp

  rootca_fp="$(json_field "${approve_json}" data.rootca_fingerprint_sha256)"
  run_logged_to_file "${output_file}" \
    nvflare package "${name}.signed.zip" --fingerprint "${rootca_fp}" --format json
}

# Process command-line arguments and check required commands.
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="initial"
WORK_DIR="${1:-./distprov_demo}"
DYNAMIC_NAME=""
DYNAMIC_YAML=""

if [[ "${1:-}" == "--add" ]]; then
  if [[ -z "${2:-}" || -z "${3:-}" ]]; then
    usage >&2
    exit 2
  fi
  MODE="add"
  DYNAMIC_NAME="$2"
  DYNAMIC_YAML="$3"
  WORK_DIR="${4:-./distprov_demo}"
fi

require_command nvflare
require_command python3

if [[ "${MODE}" == "initial" ]]; then
  # Setup the output directory. The initial demo requires a fresh work directory
  # so a run does not mix old request, signed zip, or versioned package
  # artifacts with new output.
  if [[ -e "${WORK_DIR}" ]]; then
    echo "Work directory already exists: ${WORK_DIR}" >&2
    echo "Choose a new path or remove the existing directory first." >&2
    exit 1
  fi

  umask 077
  mkdir -p "${WORK_DIR}"
else
  # Dynamic add mode is intentionally run against an existing project work
  # directory. It reuses ca/ and skips cert init.
  if [[ ! -d "${WORK_DIR}" ]]; then
    echo "Work directory does not exist: ${WORK_DIR}" >&2
    echo "Run the initial demo first, or pass the existing work directory." >&2
    exit 1
  fi
  if [[ ! -d "${WORK_DIR}/ca" ]]; then
    echo "Missing CA directory: ${WORK_DIR}/ca" >&2
    echo "Dynamic add mode requires an existing CA from the initial setup." >&2
    exit 1
  fi
fi

WORK_DIR="$(cd "${WORK_DIR}" && pwd)"

cd "${WORK_DIR}"
echo "Working directory: ${WORK_DIR}"

if [[ "${MODE}" == "add" ]]; then
  # Dynamic provisioning process for one participant added after project start.
  show "Dynamic add: requester creates request zip"
  request_participant "${DYNAMIC_NAME}" "${DYNAMIC_YAML}" "06_dynamic_request_${DYNAMIC_NAME}.json"

  show "Dynamic add: local demo handoff copies request zip"
  run_logged cp "${DYNAMIC_NAME}/${DYNAMIC_NAME}.request.zip" .

  show "Dynamic add: Project Admin approves request zip"
  approve_participant "${DYNAMIC_NAME}" "07_dynamic_approve_${DYNAMIC_NAME}.json"

  show "Dynamic add: requester packages startup kit"
  package_participant "${DYNAMIC_NAME}" "07_dynamic_approve_${DYNAMIC_NAME}.json" "08_dynamic_package_${DYNAMIC_NAME}.json"

  run_logged_to_file 09_dynamic_startup_dirs.txt \
    find workspace/fed_project -path "*/startup" -type d

  show "Result"
  echo "Startup kits:"
  sort 09_dynamic_startup_dirs.txt
  echo
  echo "Done. New participant kit generated under: ${WORK_DIR}/workspace"
  exit 0
fi

# Setup step: Project Admin creates the CA material needed to approve requests.
show "Project Admin initializes CA"
run_logged_to_file 01_cert_init.json \
  nvflare cert init --profile "${BASE_DIR}/project_profile.yaml" -o ca --version "${PROVISION_VERSION}" --force --format json

# Start of the automated distributed provisioning process.
#
# Requester side: each participant creates a request zip. The local request
# folder contains the private key and stays with that requester.
show "Requesters create request zips"
for item in "${PARTICIPANTS[@]}"; do
  name="${item%%:*}"
  yaml="${item#*:}"
  request_participant "${name}" "${yaml}" "02_request_${name}.json"
done

# Local demo handoff: copy only request zips into the Project Admin's working
# location. This models the artifact transfer; private keys remain in the
# participant request folders.
show "Local demo handoff copies request zips"
for item in "${PARTICIPANTS[@]}"; do
  name="${item%%:*}"
  run_logged cp "${name}/${name}.request.zip" .
done

# Project Admin side: approve each received request zip with the CA and project
# profile. Approval writes the approved server endpoint into the signed zip,
# then the signed zip is returned to the matching requester.
show "Project Admin approves request zips"
for item in "${PARTICIPANTS[@]}"; do
  name="${item%%:*}"
  approve_participant "${name}" "03_approve_${name}.json"
done

# Requester side: package each startup kit. Automation passes the root CA
# fingerprint from the approval output instead of prompting a human.
show "Requesters package startup kits"
for item in "${PARTICIPANTS[@]}"; do
  name="${item%%:*}"
  package_participant "${name}" "03_approve_${name}.json" "04_package_${name}.json"
done

# Summarize the generated startup-kit directories for quick inspection.
run_logged_to_file 05_startup_dirs.txt \
  find workspace/fed_project -path "*/startup" -type d

show "Result"
echo "Startup kits:"
sort 05_startup_dirs.txt
echo
echo "Done. Inspect JSON command outputs and generated kits under: ${WORK_DIR}"
