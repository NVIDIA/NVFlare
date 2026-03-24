#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#SBATCH --job-name=vlm-fl
#SBATCH --output=logs/vlm-fl-%j.out
#SBATCH --error=logs/vlm-fl-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#
# Adjust the above to match your HPC's partition and resource names.
# Some HPCs require:  #SBATCH --partition=gpu
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm_run.sh blip_vqa    # or: januspro
#

set -euo pipefail

MODEL_BACKEND="${1:-blip_vqa}"

# ---- Load modules (edit to match your HPC) ----
module purge
module load anaconda3
module load cuda/12.1    # adjust version

# ---- Activate the right conda env ----
if [[ "${MODEL_BACKEND}" == "januspro" ]]; then
    conda activate nvflare_januspro
else
    conda activate nvflare_blip
fi

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Backend:  ${MODEL_BACKEND}"
echo "Python:   $(which python)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd "${SLURM_SUBMIT_DIR}"

# ---- Step 1: Centralized baseline ----
echo ">>> Running centralized baseline ..."
python src/local_train.py \
    --model_backend "${MODEL_BACKEND}" \
    --batch_size 8 \
    --grad_accum 8 \
    --num_epochs 1 \
    --max_train_samples 2000 \
    --max_eval_samples 500 \
    --data_path "/scratch/${USER}/hf_cache" \
    --output_dir "/scratch/${USER}/${MODEL_BACKEND}_centralized"

echo ""

# ---- Step 2: FL simulator ----
echo ">>> Running FL simulation (2 clients, 3 rounds) ..."
python job.py \
    --model_backend "${MODEL_BACKEND}" \
    --simulator \
    --num_clients 2 \
    --num_rounds 3 \
    --batch_size 8 \
    --grad_accum 8 \
    --max_train_samples 2000 \
    --max_eval_samples 500 \
    --data_path "/scratch/${USER}/hf_cache"

echo ""
echo "=== Done ==="
