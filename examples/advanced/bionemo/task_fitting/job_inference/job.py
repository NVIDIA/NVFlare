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

"""
Federated Embedding Inference using FedAvgRecipe

This script replaces the launcher-based approach in task_fitting.ipynb with FedAvgRecipe.
Run this after data splitting to perform federated embedding extraction.
"""

import os

from bionemo.core.data.load import load

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

# Configuration
n_clients = 3
data_root = "/tmp/data/mixed_soft"
results_path = "/tmp/data/mixed_soft/results"
micro_bs = 64

# Download model checkpoint
checkpoint_path = load("esm2/8m:2.0")
print(f"Downloaded model to {checkpoint_path}")

# Create results directory
os.makedirs(results_path, exist_ok=True)

# Build script arguments for inference (same for all clients, paths resolved in client script)
script_args = f"--checkpoint-path {checkpoint_path} --data-root {data_root} --results-path {results_path} --precision bf16-mixed --micro-batch-size {micro_bs} --num-gpus 1"

# Create FedAvgRecipe for inference (1 round only as this is just inference)
recipe = FedAvgRecipe(
    name="esm2_embeddings",
    min_clients=n_clients,
    num_rounds=1,  # Inference only needs 1 round
    train_script="client.py",
    train_args=script_args,
)

# Run simulation
env = SimEnv(num_clients=n_clients, workspace_root="/tmp/nvflare/bionemo/embeddings", gpu_config="0", num_threads=1)
run = recipe.execute(env)
print()
print("Job Status:", run.get_status())
print("Results:", run.get_result())
print()
