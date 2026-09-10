#!/usr/bin/env bash
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

set -Eeuo pipefail

IMAGE_TAG="nvcr.io/nvidia/nemo-automodel:26.04"
SOURCE_DIR="${CHECKOUT_DIR:-$(git rev-parse --show-toplevel)}"
TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/scratch/hroth/Code/nvflare/nemotron-nano-regression-${TIMESTAMP}}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_ROOT}/cache/huggingface}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/scratch/hroth/Code/nvflare/nemotron-peft-h100-20260604-113217/integration/nemo/examples/peft/data/FinancialPhraseBank-v1.0}"
GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2n | head -1 | cut -d, -f1 | tr -d ' ')}"

mkdir -p "${RUN_ROOT}"/{artifacts,data,logs,workspace} "${CACHE_ROOT}"
printf '%s\n' "${GPU_ID}" >"${RUN_ROOT}/artifacts/selected_gpu.txt"
printf '%s\n' "${CACHE_ROOT}" >"${RUN_ROOT}/artifacts/cache_root.txt"
cp -a "${SOURCE_DATA_DIR}/." "${RUN_ROOT}/data/"
exec > >(tee -a "${RUN_ROOT}/logs/nano_runner.log") 2>&1
set -x

record_exit() {
    status=$?
    trap - EXIT
    printf '%s\n' "${status}" >"${RUN_ROOT}/artifacts/nano_regression_exit_code.txt"
    exit "${status}"
}
trap record_exit EXIT

if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable; the Nano regression was not started." >"${RUN_ROOT}/artifacts/BLOCKED.txt"
    exit 2
fi
docker pull "${IMAGE_TAG}"
IMAGE_DIGEST="$(docker image inspect "${IMAGE_TAG}" --format '{{index .RepoDigests 0}}')"
printf '%s\n' "${IMAGE_DIGEST}" >"${RUN_ROOT}/artifacts/container_digest.txt"

docker run --rm --gpus "device=${GPU_ID}" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN -e HF_HOME=/hf_cache -v "${SOURCE_DIR}:/workspace" -v "${RUN_ROOT}:/host_out" \
    -v "${CACHE_ROOT}:/hf_cache" -w /workspace/integration/nemo/examples/peft \
    "${IMAGE_DIGEST}" bash -lc '
set -Eeuo pipefail
git config --global --add safe.directory /workspace || true
NVFL_BASE_VERSION=2.10.0 python -m pip install -e /workspace
python data/split_financial_phrase_data.py --alpha=10.0 --random_seed=0 --num_clients=2 --remove_train_overlap \
  --data_path=/host_out/data/financial_phrase_bank_train.jsonl \
  --validation_path=/host_out/data/financial_phrase_bank_val.jsonl \
  --test_path=/host_out/data/financial_phrase_bank_test.jsonl --out_dir=/host_out/data_split
python prepare_initial_adapter.py --model_profile=nano --output=/host_out/artifacts/initial_adapter.pt
python job.py --model_profile=nano --n_clients=2 --num_rounds=2 --num_threads=1 --gpu="[0]" \
  --max_steps=2 --seq_length=128 --train_split_dir=/host_out/data_split \
  --validation_file=/host_out/data/financial_phrase_bank_val.jsonl \
  --workspace=/host_out/workspace --initial_adapter_ckpt=/host_out/artifacts/initial_adapter.pt
python verify_federated_run.py --client_work_dir=/host_out/workspace/automodel_work \
  --server_root=/host_out/workspace --num_clients=2 --num_rounds=2 \
  --output=/host_out/artifacts/continuity.json
FINAL=$(find /host_out/workspace -path '*/server_rounds/round_1/FL_global_model.pt' -print -quit)
test -n "${FINAL}"
python predict_sentiment.py --model_profile=nano --server_model="${FINAL}" \
  --output_dir=/host_out/artifacts/final_adapter --output_json=/host_out/artifacts/predictions.json
test -s /host_out/artifacts/predictions.json
'
