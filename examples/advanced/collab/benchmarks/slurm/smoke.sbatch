#!/bin/bash
#SBATCH --account=coreai_edgeai_flresearch
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=collab-llm-smoke
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.err

set -euo pipefail

: "${NVFLARE_SOURCE_ROOT:?Export NVFLARE_SOURCE_ROOT when submitting}"
: "${EXPECTED_COMMIT:?Export EXPECTED_COMMIT when submitting}"
: "${SCHEME:?Export SCHEME=standard or SCHEME=collab when submitting}"
if [[ "${SCHEME}" != "standard" && "${SCHEME}" != "collab" ]]; then
    echo "SCHEME must be standard or collab" >&2
    exit 1
fi
SMOKE_KIND=${SMOKE_KIND:-single}
case "${SMOKE_KIND}" in
    single)
        CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_smoke.json
        CPU_THREADS=8
        ;;
    capacity)
        CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_capacity.json
        CPU_THREADS=2
        ;;
    *) echo "SMOKE_KIND must be single or capacity" >&2; exit 1 ;;
esac

CONDA_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/miniconda3
HF_CACHE_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/huggingface_cache
RESULT_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/results/smoke_${SMOKE_KIND}_${SCHEME}_${SLURM_JOB_ID}

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"

mkdir -p "${RESULT_ROOT}/environment"
git -C "${NVFLARE_SOURCE_ROOT}" status --short --branch >"${RESULT_ROOT}/environment/git_status.txt"
git -C "${NVFLARE_SOURCE_ROOT}" rev-parse HEAD >"${RESULT_ROOT}/environment/commit.txt"
hostname >"${RESULT_ROOT}/environment/hostname.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >"${RESULT_ROOT}/environment/gpus.csv"
python -m pip freeze >"${RESULT_ROOT}/environment/packages.txt"
python -c 'import torch; assert torch.cuda.device_count() == 1; assert torch.cuda.is_bf16_supported()'
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 \
    >"${RESULT_ROOT}/environment/gpu_samples.csv" &
gpu_monitor_pid=$!
stop_gpu_monitor() {
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
}
trap stop_gpu_monitor EXIT

cd "${NVFLARE_SOURCE_ROOT}/examples/advanced"
/usr/bin/time -v -o "${RESULT_ROOT}/environment/time.txt" \
    python -m collab.benchmarks.run_benchmarks \
    --config "${CONFIG}" \
    --scheme "${SCHEME}" \
    --output-root "${RESULT_ROOT}"
stop_gpu_monitor
trap - EXIT
