#!/bin/bash
set -euo pipefail

: "${NVFLARE_SOURCE_ROOT:?Set NVFLARE_SOURCE_ROOT to the immutable collab_llm checkout}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the committed benchmark source SHA}"

CONDA_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/miniconda3
HF_CACHE_ROOT=/lustre/fsw/portfolios/coreai/users/ziyuex/huggingface_cache
CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm.json

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

cd "${NVFLARE_SOURCE_ROOT}/examples/advanced"
python collab/benchmarks/prepare_data.py --workload pt_llm_sft --config "${CONFIG}"
