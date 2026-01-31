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

"""
Federated MLP Training using FedAvgRecipe with PyTorch

This script uses PyTorch-based federated learning to train an MLP classifier 
on protein embeddings for subcellular location prediction.
Run this after inference to train an MLP classifier on the embeddings.
"""

import os

from model import CLASS_LABELS, ProteinMLP

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_experiment_tracking

# Configuration
n_clients = 3
data_root = "/tmp/data/mixed_soft"
results_path = "/tmp/data/mixed_soft/results"
split_alpha = 1.0
embedding_dimensions = 1280  # embedding dimensions of ESM2-650m (use 320 for ESM2-8m)

# Set environment variable for local vs federated training
training_mode = input("Training mode (local/fedavg): ").strip().lower()
if training_mode == "local":
    os.environ["SIM_LOCAL"] = "True"
    job_name = "mlp_local"
    print("Running LOCAL training simulation...")
else:
    os.environ["SIM_LOCAL"] = "False"
    job_name = "mlp_fedavg"
    print("Running FEDERATED training...")

# Build script arguments for MLP training
script_args = (
    f"--data-root {data_root} "
    f"--results-path {results_path} "
    f"--aggregation-epochs 20 "
    f"--lr 1e-5 "
    f"--batch-size 128 "
    f"--embedding-dimensions {embedding_dimensions}"
)

# Create FedAvgRecipe for MLP training
print(f"Creating initial model with {len(CLASS_LABELS)} classes")

recipe = FedAvgRecipe(
    name=job_name,
    min_clients=n_clients,
    num_rounds=50,
    # Model can be specified as class instance or dict config:
    initial_model=ProteinMLP(input_dim=embedding_dimensions, num_classes=len(CLASS_LABELS)),
    # Alternative: initial_model={"path": "model.ProteinMLP", "args": {...}},
    # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
    train_script="client.py",
    train_args=script_args,
)

# Enable TensorBoard tracking
add_experiment_tracking(recipe, "tensorboard")

# Run simulation
env = SimEnv(
    num_clients=n_clients,
    workspace_root=f"/tmp/nvflare/bionemo/{job_name}_alpha{split_alpha}",
    num_threads=n_clients,  # MLP training can run in parallel
)
run = recipe.execute(env)
print()
print("Job Status:", run.get_status())
print("Results:", run.get_result())
print()
print("\nView TensorBoard results with:")
print(f"tensorboard --logdir /tmp/nvflare/bionemo/{job_name}_alpha{split_alpha}")
