#!/bin/bash
set -euo pipefail

: "${NVFLARE_SOURCE_ROOT:?Export NVFLARE_SOURCE_ROOT when submitting}"
: "${EXPECTED_COMMIT:?Export EXPECTED_COMMIT when submitting}"
: "${SLURM_JOB_ID:?Run this entrypoint inside a Slurm allocation}"

RUN_KIND=${1:?Pass single, capacity, or paired}
case "${RUN_KIND}" in
    single)
        : "${SCHEME:?Export SCHEME=standard or SCHEME=collab}"
        CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_yi_9b_single.json
        EXPECTED_GPU_COUNT=1
        RESULT_NAME=yi_9b_single_${SCHEME}_${SLURM_JOB_ID}
        ;;
    capacity)
        : "${SCHEME:?Export SCHEME=standard or SCHEME=collab}"
        CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_yi_9b_capacity.json
        EXPECTED_GPU_COUNT=4
        RESULT_NAME=yi_9b_capacity_${SCHEME}_${SLURM_JOB_ID}
        ;;
    paired)
        SCHEME_ORDER=${SCHEME_ORDER:-"standard collab"}
        if [[ "${SCHEME_ORDER}" != "standard collab" && "${SCHEME_ORDER}" != "collab standard" ]]; then
            echo "SCHEME_ORDER must be 'standard collab' or 'collab standard'" >&2
            exit 1
        fi
        CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_yi_9b.json
        EXPECTED_GPU_COUNT=4
        RESULT_NAME=yi_9b_paired_${SLURM_JOB_ID}
        ;;
    *) echo "RUN_KIND must be single, capacity, or paired" >&2; exit 1 ;;
esac
if [[ "${RUN_KIND}" != "paired" && "${SCHEME:-}" != "standard" && "${SCHEME:-}" != "collab" ]]; then
    echo "SCHEME must be standard or collab" >&2
    exit 1
fi

CONDA_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/miniconda3
HF_CACHE_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/huggingface_cache
RESULT_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/results/${RESULT_NAME}

actual_commit=$(git -C "${NVFLARE_SOURCE_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Source commit mismatch: expected ${EXPECTED_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${NVFLARE_SOURCE_ROOT}" status --porcelain)" ]]; then
    echo "Source checkout is not clean" >&2
    exit 1
fi

source "${CONDA_ROOT}/bin/activate" dfkd_async
export PYTHONPATH="${NVFLARE_SOURCE_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export XDG_CACHE_HOME="${HF_CACHE_ROOT}/xdg"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "${RESULT_ROOT}/environment"
git -C "${NVFLARE_SOURCE_ROOT}" status --short --branch >"${RESULT_ROOT}/environment/git_status.txt"
git -C "${NVFLARE_SOURCE_ROOT}" rev-parse HEAD >"${RESULT_ROOT}/environment/commit.txt"
hostname >"${RESULT_ROOT}/environment/hostname.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv >"${RESULT_ROOT}/environment/gpus.csv"
python -m pip freeze >"${RESULT_ROOT}/environment/packages.txt"
python -c "import torch; assert torch.cuda.device_count() == ${EXPECTED_GPU_COUNT}; assert torch.cuda.is_bf16_supported()"

cd "${NVFLARE_SOURCE_ROOT}/examples/advanced"
python collab/benchmarks/prepare_model.py --config "${CONFIG}" --local-files-only \
    >"${RESULT_ROOT}/environment/model_validation.txt"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 \
    >"${RESULT_ROOT}/environment/gpu_samples.csv" &
gpu_monitor_pid=$!
stop_gpu_monitor() {
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
}
trap stop_gpu_monitor EXIT

if [[ "${RUN_KIND}" == "paired" ]]; then
    read -r -a schemes <<<"${SCHEME_ORDER}"
    /usr/bin/time -v -o "${RESULT_ROOT}/environment/time.txt" \
        python -m collab.benchmarks.run_benchmarks \
        --config "${CONFIG}" \
        --scheme "${schemes[@]}" \
        --output-root "${RESULT_ROOT}"
else
    /usr/bin/time -v -o "${RESULT_ROOT}/environment/time.txt" \
        python -m collab.benchmarks.run_benchmarks \
        --config "${CONFIG}" \
        --scheme "${SCHEME}" \
        --output-root "${RESULT_ROOT}"
fi
stop_gpu_monitor
trap - EXIT
