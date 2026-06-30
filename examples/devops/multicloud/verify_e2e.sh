#!/usr/bin/env bash
#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVOPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NVFLARE_ROOT="${NVFLARE_ROOT:-$(cd "${DEVOPS_ROOT}/../.." && pwd)}"

E2E_HELPER="${SCRIPT_DIR}/e2e/verify.py"
E2E_JOBS_DIR="${SCRIPT_DIR}/e2e/jobs"

CONFIG="${SCRIPT_DIR}/all-clouds.yaml"
KUBECONFIG_DIR="${DEVOPS_ROOT}/.tmp/kubeconfigs"
WORKSPACE="${SCRIPT_DIR}/.work/e2e"
STARTUP_KIT=""
NVFLARE_CMD="${NVFLARE_CMD:-}"
JOB_IMAGE=""
JOB_TYPE="cifar10"
PYTHON_PATH="/usr/local/bin/python3"
STUDY="default"
DATASET="data"
DATA_ROOT=""
DOWNLOAD_ROOT="/tmp/nvflare/e2e-data"
ALLOW_DOWNLOAD=0
KEEP=0
NAMESPACE=""
TIMEOUT=1800
NUM_ROUNDS=2
NUM_CLIENTS=2
MAX_TRAIN_SAMPLES=128
MAX_VAL_SAMPLES=128
BATCH_SIZE=32
EPOCHS=1
TORCH_THREADS=1
JOB_NAME=""
JOB_ID=""
JOB_DONE=0
ABORT_ON_FAIL=1

usage() {
    cat <<'EOF'
Usage: examples/devops/multicloud/verify_e2e.sh [options]

Run a NumPy or CIFAR10 CPU FedAvg end-to-end smoke check against an existing
NVFlare Kubernetes deployment. The script does not create clusters or deploy
FLARE.

Defaults target the CIFAR10 production-style study-data PVC path:
  /data/<study>/<dataset>

Options:
  --config PATH          Multicloud deploy config YAML. Default: examples/devops/multicloud/all-clouds.yaml
  --kubeconfig-dir DIR   Directory containing <cloud>.yaml kubeconfigs. Default: <devops-repo>/.tmp/kubeconfigs
  --startup-kit PATH     Admin startup kit directory. Default: latest provisioned startup containing fed_admin.json
  --nvflare-cmd PATH     nvflare command. Default: NVFLARE_CMD, devops .venv, NVFlare source .venv, then PATH
  --job-image IMAGE      Override job container image for all sites.
                         Default: per-site parent docker_image from --config
  --job-type TYPE        Job to run: cifar10 or numpy. Default: cifar10
  --python-path PATH     Python path inside the job image. Default: /usr/local/bin/python3
  --namespace NS         Override all participant namespaces from --config
  --num-rounds N         FedAvg rounds. Default: 2
  --num-clients N        Number of clients required and targeted from config. Default: 2
  --timeout SECONDS      Job wait timeout. Default: 1800
  --workspace DIR        Local temp workspace. Default: examples/devops/multicloud/.work/e2e
  --study NAME           Study name. Default: default
  --dataset NAME         Study-data dataset name. CIFAR10 only. Default: data
  --data-root PATH       Preloaded CIFAR10 root inside job pods. CIFAR10 only.
                         Default: /data/<study>/<dataset>
  --skip-download        Require CIFAR10 to be preloaded at --data-root. This is the default.
  --download             Allow job pods to download CIFAR10 to --download-root for demo clusters.
  --download-root PATH   Data root used with --download. CIFAR10 only.
                         Default: /tmp/nvflare/e2e-data
  --max-train-samples N  Training subset per client. Default: 128
  --max-val-samples N    Validation subset per client. Default: 128
  --batch-size N         Training batch size. Default: 32
  --epochs N             Local epochs per round. Default: 1
  --torch-threads N      CPU threads used by torch in each job pod. Default: 1
  --keep                 Keep local generated job and downloaded results.
  -h, --help             Show this help.

Pass criteria:
  - FLARE parent deployments are available before submit.
  - The submitted job reaches a FINISHED/COMPLETED terminal status.
  - Downloaded results include a global model artifact.
  - Logs include one E2E round marker for every requested round.
  - Logs do not include obvious error markers.
  - Existing pods do not gain restarts, and pods created during the run have zero restarts.
EOF
}

log() {
    printf '[verify_e2e] %s\n' "$*"
}

die() {
    printf '[verify_e2e] ERROR: %s\n' "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --kubeconfig-dir)
            KUBECONFIG_DIR="$2"
            shift 2
            ;;
        --startup-kit)
            STARTUP_KIT="$2"
            shift 2
            ;;
        --nvflare-cmd)
            NVFLARE_CMD="$2"
            shift 2
            ;;
        --job-image)
            JOB_IMAGE="$2"
            shift 2
            ;;
        --job-type|--job)
            JOB_TYPE="$2"
            shift 2
            ;;
        --python-path)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --num-rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --study)
            STUDY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --skip-download)
            ALLOW_DOWNLOAD=0
            shift
            ;;
        --download|--allow-download)
            ALLOW_DOWNLOAD=1
            shift
            ;;
        --download-root)
            DOWNLOAD_ROOT="$2"
            shift 2
            ;;
        --max-train-samples)
            MAX_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --max-val-samples)
            MAX_VAL_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --torch-threads)
            TORCH_THREADS="$2"
            shift 2
            ;;
        --keep)
            KEEP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -f "${E2E_HELPER}" ]] || die "E2E helper not found: ${E2E_HELPER}"
[[ -d "${E2E_JOBS_DIR}" ]] || die "E2E job templates not found: ${E2E_JOBS_DIR}"
case "${JOB_TYPE}" in
    cifar10|numpy)
        ;;
    *)
        die "--job-type must be cifar10 or numpy"
        ;;
esac
[[ "${NUM_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || die "--num-rounds must be a positive integer"
[[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || die "--num-clients must be a positive integer"
[[ "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "--timeout must be a positive integer"
[[ "${MAX_TRAIN_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || die "--max-train-samples must be a positive integer"
[[ "${MAX_VAL_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || die "--max-val-samples must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
[[ "${TORCH_THREADS}" =~ ^[1-9][0-9]*$ ]] || die "--torch-threads must be a positive integer"

if [[ -z "${DATA_ROOT}" ]]; then
    if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
        DATA_ROOT="${DOWNLOAD_ROOT}"
    else
        DATA_ROOT="/data/${STUDY}/${DATASET}"
    fi
fi

if [[ -z "${NVFLARE_CMD}" && -x "${DEVOPS_ROOT}/.venv/bin/nvflare" ]]; then
    NVFLARE_CMD="${DEVOPS_ROOT}/.venv/bin/nvflare"
elif [[ -z "${NVFLARE_CMD}" && -n "${NVFLARE_ROOT}" && -x "${NVFLARE_ROOT}/.venv/bin/nvflare" ]]; then
    NVFLARE_CMD="${NVFLARE_ROOT}/.venv/bin/nvflare"
fi
if [[ -z "${NVFLARE_CMD}" ]]; then
    NVFLARE_CMD="$(command -v nvflare || true)"
fi
[[ -n "${NVFLARE_CMD}" ]] || die "nvflare command not found. Install NVFlare or create .venv/bin/nvflare."
[[ -x "${NVFLARE_CMD}" ]] || die "nvflare command is not executable: ${NVFLARE_CMD}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
"${PYTHON_BIN}" "${E2E_HELPER}" --help >/dev/null 2>&1 \
    || die "Python dependencies for ${E2E_HELPER} are missing. Install PyYAML in ${PYTHON_BIN} or run from the deploy environment."

if [[ -z "${STARTUP_KIT}" ]]; then
    ADMIN_CONFIG="$(
        find "${SCRIPT_DIR}/.work/provision" -path '*/prod_*/*/startup/fed_admin.json' -type f 2>/dev/null \
            | sort \
            | tail -n 1 || true
    )"
    if [[ -n "${ADMIN_CONFIG}" ]]; then
        STARTUP_KIT="${ADMIN_CONFIG%/fed_admin.json}"
    fi
fi
[[ -n "${STARTUP_KIT}" && -d "${STARTUP_KIT}" ]] || die "admin startup kit not found. Pass --startup-kit."

mkdir -p "${WORKSPACE}"
PARTICIPANTS_TSV="${WORKSPACE}/participants.tsv"
SELECTED_CLIENTS="${WORKSPACE}/selected_clients.txt"

load_participants() {
    local cmd=(
        "${PYTHON_BIN}" "${E2E_HELPER}" participants
        --config "${CONFIG}"
        --kubeconfig-dir "${KUBECONFIG_DIR}"
        --num-clients "${NUM_CLIENTS}"
        --job-type "${JOB_TYPE}"
        --study "${STUDY}"
        --dataset "${DATASET}"
        --data-root "${DATA_ROOT}"
    )
    [[ -n "${NAMESPACE}" ]] && cmd+=(--namespace "${NAMESPACE}")
    [[ -n "${JOB_IMAGE}" ]] && cmd+=(--job-image "${JOB_IMAGE}")
    [[ "${ALLOW_DOWNLOAD}" == "1" ]] && cmd+=(--allow-download)
    "${cmd[@]}" > "${PARTICIPANTS_TSV}"
}

load_participants

JOB_IMAGE="$(awk -F '\t' 'NR == 1 {print $6}' "${PARTICIPANTS_TSV}")"
awk -F '\t' '$1 != "server" {print $2}' "${PARTICIPANTS_TSV}" > "${SELECTED_CLIENTS}"

cleanup() {
    local exit_code=$?
    if [[ "${exit_code}" -ne 0 && -n "${JOB_ID}" && "${JOB_DONE}" != "1" && "${ABORT_ON_FAIL}" == "1" ]]; then
        log "aborting unfinished job ${JOB_ID}"
        "${NVFLARE_CMD}" job abort "${JOB_ID}" --startup-kit "${STARTUP_KIT}" --study "${STUDY}" --force >/dev/null 2>&1 || true
    fi
    if [[ "${KEEP}" != "1" && -d "${WORKSPACE}" ]]; then
        rm -rf "${WORKSPACE}"
    elif [[ -d "${WORKSPACE}" ]]; then
        log "kept workspace: ${WORKSPACE}"
    fi
}
trap cleanup EXIT

verify_deployments_available() {
    log "checking FLARE parent deployments"
    while IFS=$'\t' read -r _role name namespace kubeconfig _cloud _image; do
        [[ -f "${kubeconfig}" ]] || die "kubeconfig not found for ${name}: ${kubeconfig}"
        kubectl --kubeconfig "${kubeconfig}" -n "${namespace}" wait \
            --for=condition=available "deployment/${name}" --timeout=600s
    done < "${PARTICIPANTS_TSV}"
}

capture_restarts() {
    local output="$1"
    : > "${output}"
    while IFS=$'\t' read -r _role name namespace kubeconfig _cloud _image; do
        kubectl --kubeconfig "${kubeconfig}" -n "${namespace}" get pods -o json \
            | "${PYTHON_BIN}" "${E2E_HELPER}" pod-restarts --participant "${name}" --namespace "${namespace}" \
            >> "${output}"
    done < "${PARTICIPANTS_TSV}"
}

compare_restarts() {
    "${PYTHON_BIN}" "${E2E_HELPER}" compare-restarts "$1" "$2"
}

collect_k8s_job_logs() {
    local output="$1"
    : > "${output}"
    while IFS=$'\t' read -r _role _name namespace kubeconfig _cloud _image; do
        while IFS= read -r pod; do
            [[ -n "${pod}" ]] || continue
            {
                printf '===== %s/%s =====\n' "${namespace}" "${pod}"
                kubectl --kubeconfig "${kubeconfig}" -n "${namespace}" logs "${pod}" --tail=2000 2>&1 || true
            } >> "${output}"
        done < <(
            kubectl --kubeconfig "${kubeconfig}" -n "${namespace}" get pods -o json \
                | "${PYTHON_BIN}" "${E2E_HELPER}" list-job-pods --job-id "${JOB_ID}"
        )
    done < "${PARTICIPANTS_TSV}"
}

create_job() {
    local job_dir="$1"
    case "${JOB_TYPE}" in
        cifar10)
            JOB_NAME="cifar10_cpu_fedavg_e2e_$(date +%Y%m%d_%H%M%S)"
            ;;
        numpy)
            JOB_NAME="numpy_fedavg_e2e_$(date +%Y%m%d_%H%M%S)"
            ;;
    esac

    local cmd=(
        "${PYTHON_BIN}" "${E2E_HELPER}" create-job
        --job-dir "${job_dir}"
        --job-name "${JOB_NAME}"
        --job-type "${JOB_TYPE}"
        --job-image "${JOB_IMAGE}"
        --python-path "${PYTHON_PATH}"
        --num-rounds "${NUM_ROUNDS}"
        --num-clients "${NUM_CLIENTS}"
        --data-root "${DATA_ROOT}"
        --max-train-samples "${MAX_TRAIN_SAMPLES}"
        --max-val-samples "${MAX_VAL_SAMPLES}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${EPOCHS}"
        --torch-threads "${TORCH_THREADS}"
        --selected-clients "${SELECTED_CLIENTS}"
        --participants-tsv "${PARTICIPANTS_TSV}"
        --templates-dir "${E2E_JOBS_DIR}"
    )
    [[ "${ALLOW_DOWNLOAD}" == "1" ]] && cmd+=(--allow-download)
    "${cmd[@]}"
}

json_get() {
    "${PYTHON_BIN}" "${E2E_HELPER}" json-get "$1"
}

validate_download() {
    "${PYTHON_BIN}" "${E2E_HELPER}" validate-result \
        --download-json "$1" \
        --logs-json "$2" \
        --k8s-logs "$3" \
        --num-rounds "${NUM_ROUNDS}" \
        --job-type "${JOB_TYPE}"
}

verify_deployments_available
capture_restarts "${WORKSPACE}/restarts.before.tsv"

JOB_DIR="${WORKSPACE}/job"
DOWNLOAD_DIR="${WORKSPACE}/download"
mkdir -p "${DOWNLOAD_DIR}"
create_job "${JOB_DIR}"

log "using job images from launcher_spec:"
awk -F '\t' '{printf "  %s: %s\n", $2, $6}' "${PARTICIPANTS_TSV}"
log "job type: ${JOB_TYPE}"
if [[ "${JOB_TYPE}" == "numpy" ]]; then
    log "data mode: numpy smoke job does not require dataset mounts or downloads"
elif [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
    log "data mode: job pod download to ${DATA_ROOT}"
else
    log "data mode: preloaded study-data PVC at ${DATA_ROOT}"
fi

log "submitting ${JOB_NAME}"
SUBMIT_JSON="${WORKSPACE}/submit.json"
"${NVFLARE_CMD}" job submit -j "${JOB_DIR}" --startup-kit "${STARTUP_KIT}" --study "${STUDY}" --format json \
    | tee "${SUBMIT_JSON}"
JOB_ID="$(json_get data.job_id < "${SUBMIT_JSON}")"
[[ -n "${JOB_ID}" ]] || die "failed to parse submitted job id"
log "submitted job_id=${JOB_ID}"

WAIT_JSON="${WORKSPACE}/wait.json"
log "waiting up to ${TIMEOUT}s for job completion"
"${NVFLARE_CMD}" job wait "${JOB_ID}" --startup-kit "${STARTUP_KIT}" --study "${STUDY}" \
    --timeout "${TIMEOUT}" --interval 5 --format json | tee "${WAIT_JSON}"
JOB_DONE=1
STATUS="$(json_get data.status < "${WAIT_JSON}")"
case "${STATUS}" in
    FINISHED:COMPLETED|FINISHED_COMPLETED|FINISHED_OK|FINISHED)
        ;;
    *)
        die "unexpected job status: ${STATUS}"
        ;;
esac
log "job completed with status ${STATUS}"

DOWNLOAD_JSON="${WORKSPACE}/download.json"
log "downloading job result"
"${NVFLARE_CMD}" job download "${JOB_ID}" --startup-kit "${STARTUP_KIT}" --study "${STUDY}" \
    -o "${DOWNLOAD_DIR}" --force --format json | tee "${DOWNLOAD_JSON}"

LOGS_JSON="${WORKSPACE}/logs.json"
log "fetching job logs"
"${NVFLARE_CMD}" job logs "${JOB_ID}" --startup-kit "${STARTUP_KIT}" --study "${STUDY}" \
    --site all --tail 2000 --format json > "${LOGS_JSON}"

K8S_LOGS="${WORKSPACE}/k8s_job_pod_logs.txt"
log "collecting Kubernetes job pod logs"
collect_k8s_job_logs "${K8S_LOGS}"

validate_download "${DOWNLOAD_JSON}" "${LOGS_JSON}" "${K8S_LOGS}"

capture_restarts "${WORKSPACE}/restarts.after.tsv"
compare_restarts "${WORKSPACE}/restarts.before.tsv" "${WORKSPACE}/restarts.after.tsv"

log "PASS: Kubernetes ${JOB_TYPE} CPU FedAvg E2E verification succeeded for job ${JOB_ID}"
