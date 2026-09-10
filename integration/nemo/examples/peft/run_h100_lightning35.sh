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

IMAGE_TAG="nvcr.io/nvidia/nemo-automodel:26.08"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
SOURCE_DIR="${CHECKOUT_DIR:-$(git rev-parse --show-toplevel)}"
TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/scratch/hroth/Code/nvflare/nemotron35-lightning-${TIMESTAMP}}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_ROOT}/cache/huggingface}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/scratch/hroth/Code/nvflare/nemotron-peft-h100-20260604-113217/integration/nemo/examples/peft/data/FinancialPhraseBank-v1.0}"
CONTAINER_NAME="nvflare-lightning35-${TIMESTAMP,,}"
EXAMPLE_DIR="/workspace/integration/nemo/examples/peft"
TELEMETRY_PID=""

mkdir -p "${RUN_ROOT}"/{artifacts,data,logs,runs,evaluation} "${CACHE_ROOT}"
exec > >(tee -a "${RUN_ROOT}/logs/host_runner.log") 2>&1
set -x
printf '%s\n' "${CACHE_ROOT}" >"${RUN_ROOT}/artifacts/cache_root.txt"
git -C "${SOURCE_DIR}" rev-parse HEAD >"${RUN_ROOT}/artifacts/commit_sha.txt"
git -C "${SOURCE_DIR}" status --short >"${RUN_ROOT}/artifacts/git_status.txt"
uname -a >"${RUN_ROOT}/artifacts/uname.txt"
id >"${RUN_ROOT}/artifacts/user_identity.txt"
free -h >"${RUN_ROOT}/artifacts/host_memory.txt"
ls -l /var/run/docker.sock >"${RUN_ROOT}/artifacts/docker_socket.txt" 2>&1 || true
nvidia-smi -L >"${RUN_ROOT}/artifacts/nvidia_smi_L.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv \
    >"${RUN_ROOT}/artifacts/gpu_preflight.csv"
printf 'stage,start_epoch,end_epoch,elapsed_seconds,exit_code\n' >"${RUN_ROOT}/artifacts/stage_timings.csv"

cleanup() {
    status=$?
    if [[ -n "${TELEMETRY_PID}" ]]; then
        kill "${TELEMETRY_PID}" 2>/dev/null || true
    fi
    docker logs "${CONTAINER_NAME}" >"${RUN_ROOT}/logs/container.log" 2>&1 || true
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    printf '%s\n' "${status}" >"${RUN_ROOT}/artifacts/runner_exit_code.txt"
    python3 "${SOURCE_DIR}/integration/nemo/examples/peft/render_validation_report.py" \
        --run_root "${RUN_ROOT}" --exit_code "${status}" || true
    exit "${status}"
}
trap cleanup EXIT

if ! docker info >"${RUN_ROOT}/logs/docker_info.txt" 2>&1; then
    cat >"${RUN_ROOT}/artifacts/BLOCKED.txt" <<'EOF'
Docker is unavailable to this user. Restore normal access to /var/run/docker.sock through the host administrator,
then rerun this script. The runner never changes Docker socket ownership or permissions.
EOF
    exit 2
fi

if [[ ! -f "${SOURCE_DATA_DIR}/financial_phrase_bank_train.jsonl" ]]; then
    echo "Financial PhraseBank input is missing from ${SOURCE_DATA_DIR}" >&2
    exit 2
fi
cp -a "${SOURCE_DATA_DIR}/." "${RUN_ROOT}/data/"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2n | head -1 | cut -d, -f1 | tr -d ' ')}"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 10 \
    >"${RUN_ROOT}/logs/gpu_telemetry.csv" &
TELEMETRY_PID=$!

docker pull "${IMAGE_TAG}"
IMAGE_DIGEST="$(docker image inspect "${IMAGE_TAG}" --format '{{index .RepoDigests 0}}')"
printf '%s\n' "${IMAGE_DIGEST}" >"${RUN_ROOT}/artifacts/container_digest.txt"

docker run -d --name "${CONTAINER_NAME}" --gpus "device=${GPU_ID}" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache \
    -v "${SOURCE_DIR}:/workspace" -v "${RUN_ROOT}:/host_out" \
    -v "${CACHE_ROOT}:/hf_cache" \
    -w "${EXAMPLE_DIR}" "${IMAGE_DIGEST}" sleep infinity

run_stage() {
    stage="$1"
    shift
    stage_start="$(date +%s)"
    set +e
    docker exec "${CONTAINER_NAME}" bash -lc "$*" 2>&1 | tee "${RUN_ROOT}/logs/${stage}.log"
    stage_status="${PIPESTATUS[0]}"
    set -e
    stage_end="$(date +%s)"
    printf '%s\n' "${stage_status}" >"${RUN_ROOT}/artifacts/${stage}_exit_code.txt"
    printf '%s,%s,%s,%s,%s\n' "${stage}" "${stage_start}" "${stage_end}" "$((stage_end - stage_start))" \
        "${stage_status}" >>"${RUN_ROOT}/artifacts/stage_timings.csv"
    return "${stage_status}"
}

run_stage setup "git config --global --add safe.directory /workspace || true
NVFL_BASE_VERSION=2.10.0 python -m pip install -e /workspace && python - <<'PY'
import importlib.metadata as m
import json
import torch
import nvflare
import nemo_automodel
assert torch.cuda.is_available()
info = {
    'cuda_device': torch.cuda.get_device_name(0),
    'torch': torch.__version__,
    'nvflare': nvflare.__version__,
    'nemo_automodel': m.version('nemo-automodel'),
}
print(json.dumps(info, indent=2))
PY"

run_stage resolve_model "python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
model_id = '${MODEL_ID}'
revision = HfApi().model_info(model_id).sha
print(revision)
snapshot_download(model_id, revision=revision, cache_dir='/hf_cache')
open('/host_out/artifacts/model_revision.txt', 'w').write(revision + '\n')
open('/host_out/artifacts/tokenizer_revision.txt', 'w').write(revision + '\n')
PY"
MODEL_REVISION="$(cat "${RUN_ROOT}/artifacts/model_revision.txt")"

IDENTITY="--model_profile=lightning35 --model_name_or_path=${MODEL_ID} --tokenizer_name_or_path=${MODEL_ID} --model_revision=${MODEL_REVISION} --tokenizer_revision=${MODEL_REVISION} --lora_rank=8 --lora_alpha=32 --lora_dropout=0 --target_modules=all-linear --exclude_modules='*.out_proj' --use_triton_lora --tp_size=1 --cp_size=1 --ep_size=1"
TRAIN_COMMON="${IDENTITY} --seq_length=512 --micro_batch_size=1 --global_batch_size=1 --learning_rate=5e-5"
DATA_ARGS="--train_split_dir=/host_out/data_split --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --alpha=10.0"

run_stage data "python data/split_financial_phrase_data.py --alpha=10.0 --random_seed=0 --num_clients=3 --remove_train_overlap --data_path=/host_out/data/financial_phrase_bank_train.jsonl --validation_path=/host_out/data/financial_phrase_bank_val.jsonl --test_path=/host_out/data/financial_phrase_bank_test.jsonl --out_dir=/host_out/data_split"
run_stage cpu_init "python - <<'PY'
import adapter_checkpoint
import job
import model_profiles
import torch
from types import SimpleNamespace
state={'model.test.lora_A.weight': torch.zeros((2, 2), dtype=torch.float32)}
plain=adapter_checkpoint.strip_model_prefix(state)
profile=SimpleNamespace(model_profile='lightning35', **model_profiles.profile_defaults('lightning35'))
manifest=adapter_checkpoint.build_adapter_manifest(
    plain,
    model_profile='lightning35',
    model_name_or_path='${MODEL_ID}',
    tokenizer_name_or_path='${MODEL_ID}',
    model_revision='${MODEL_REVISION}',
    tokenizer_revision='${MODEL_REVISION}',
    profile_settings=job._profile_settings(profile),
)
adapter_checkpoint.save_nvflare_adapter_checkpoint(state, '/host_out/artifacts/cpu_initial_adapter.pt', adapter_manifest=manifest)
PY"
run_stage cpu_federation "python job.py ${TRAIN_COMMON} --backend=mock --n_clients=3 --num_rounds=3 --num_threads=1 --max_steps=1 --mock_site_steps=1,2,4 --mock_site_deltas=0.1,0.2,0.4 --workspace=/host_out/runs/cpu_federation/workspace --initial_adapter_ckpt=/host_out/artifacts/cpu_initial_adapter.pt"
run_stage cpu_federation_verify "python verify_federated_run.py --client_work_dir=/host_out/runs/cpu_federation/workspace/automodel_work --server_root=/host_out/runs/cpu_federation/workspace --num_clients=3 --num_rounds=3 --output=/host_out/runs/cpu_federation/continuity.json"
run_stage initialize "python prepare_initial_adapter.py ${IDENTITY} --seed=42 --output=/host_out/artifacts/initial_adapter.pt"
run_stage base_eval "python evaluate_sentiment.py ${IDENTITY} --no-search_validation_bias --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --test_file=/host_out/data/financial_phrase_bank_test.jsonl --output_dir=/host_out/evaluation/base"

run_federation() {
    label="$1"
    clients="$2"
    rounds="$3"
    steps="$4"
    seed="$5"
    activation="$6"
    workspace="/host_out/runs/${label}/workspace"
    run_stage "${label}" "python job.py ${TRAIN_COMMON} ${DATA_ARGS} --seed=${seed} --n_clients=${clients} --num_rounds=${rounds} --num_threads=1 --gpu='[0]' --max_steps=${steps} --workspace=${workspace} --initial_adapter_ckpt=/host_out/artifacts/initial_adapter.pt ${activation}" || return $?
    run_stage "${label}_continuity" "python verify_federated_run.py --client_work_dir=${workspace}/automodel_work --server_root=${workspace} --num_clients=${clients} --num_rounds=${rounds} --output=/host_out/runs/${label}/continuity.json" || return $?
}

SMOKE_LABEL="smoke"
if ! run_federation "${SMOKE_LABEL}" 1 1 2 42 "--no-activation_checkpointing"; then
    SMOKE_LABEL="smoke_activation_checkpointing"
    if ! run_federation "${SMOKE_LABEL}" 1 1 2 42 "--activation_checkpointing"; then
        echo "Lightning LoRA is not feasible on one H100 for this workload, including activation checkpointing." \
            >"${RUN_ROOT}/artifacts/SINGLE_GPU_FEASIBILITY_FAILURE.txt"
        exit 3
    fi
    ACTIVATION_ARG="--activation_checkpointing"
else
    ACTIVATION_ARG="--no-activation_checkpointing"
fi

run_stage smoke_eval_a "FINAL=\$(find /host_out/runs/${SMOKE_LABEL}/workspace -path '*/server_rounds/round_0/FL_global_model.pt' -print -quit); python evaluate_sentiment.py ${IDENTITY} --no-search_validation_bias --validation_only --adapter_dir=\${FINAL} --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --output_dir=/host_out/evaluation/smoke_a"
run_stage smoke_eval_b "FINAL=\$(find /host_out/runs/${SMOKE_LABEL}/workspace -path '*/server_rounds/round_0/FL_global_model.pt' -print -quit); python evaluate_sentiment.py ${IDENTITY} --no-search_validation_bias --validation_only --adapter_dir=\${FINAL} --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --output_dir=/host_out/evaluation/smoke_b"
run_stage smoke_reload_compare "python - <<'PY'
import json
a=json.load(open('/host_out/evaluation/smoke_a/summary.json'))
b=json.load(open('/host_out/evaluation/smoke_b/summary.json'))
for split in ('validation',):
    for metric in ('response_token_loss','accuracy','macro_f1','confusion','prediction_counts'):
        assert a[split][metric] == b[split][metric], (split, metric)
print('native reload scores match')
PY"

run_federation continuity 3 3 2 42 "${ACTIVATION_ARG}"

for seed in 42 43; do
    label="learning_seed${seed}"
    run_federation "${label}" 3 3 300 "${seed}" "${ACTIVATION_ARG}"
    for round in 0 1 2; do
        run_stage "${label}_round${round}_val" "CKPT=\$(find /host_out/runs/${label}/workspace -path \"*/server_rounds/round_${round}/FL_global_model.pt\" -print -quit); python evaluate_sentiment.py ${IDENTITY} --no-search_validation_bias --validation_only --adapter_dir=\${CKPT} --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --output_dir=/host_out/evaluation/${label}/round_${round}_validation"
    done
    run_stage "${label}_final_test" "CKPT=\$(find /host_out/runs/${label}/workspace -path '*/server_rounds/round_2/FL_global_model.pt' -print -quit); python evaluate_sentiment.py ${IDENTITY} --no-search_validation_bias --adapter_dir=\${CKPT} --validation_file=/host_out/data/financial_phrase_bank_val.jsonl --test_file=/host_out/data/financial_phrase_bank_test.jsonl --output_dir=/host_out/evaluation/${label}/final"
done

run_stage acceptance "python assess_validation.py --base_summary=/host_out/evaluation/base/summary.json --smoke_client_root=/host_out/runs/${SMOKE_LABEL}/workspace/automodel_work --continuity_report=/host_out/runs/continuity/continuity.json --seed_summary=/host_out/evaluation/learning_seed42/final/summary.json --seed_summary=/host_out/evaluation/learning_seed43/final/summary.json --output=/host_out/artifacts/acceptance.json"
run_stage provenance "git -C /workspace rev-parse HEAD > /host_out/artifacts/commit_sha.txt && python -m pip freeze > /host_out/artifacts/pip_freeze.txt && cp /host_out/runs/learning_seed42/workspace/automodel_work/site-1/site-1_round_0/finetune_config.yaml /host_out/artifacts/resolved_training_config.yaml"

if [[ "${RUN_NANO_REGRESSION:-1}" == "1" ]]; then
    CHECKOUT_DIR="${SOURCE_DIR}" RUN_ROOT="${RUN_ROOT}/nano_regression" SOURCE_DATA_DIR="${RUN_ROOT}/data" \
        GPU_ID="${GPU_ID}" "${SOURCE_DIR}/integration/nemo/examples/peft/run_h100_nano_regression.sh"
fi

cat >"${RUN_ROOT}/artifacts/SCOPE.txt" <<'EOF'
This run validates sequential three-client federated simulation on one H100. It does not validate separate-host
deployment or distributed training within one client.
EOF
