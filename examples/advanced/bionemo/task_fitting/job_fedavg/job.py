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
Federated MLP Training using FedAvgRecipe

This script replaces the ModelLearner-based approach in task_fitting.ipynb with FedAvgRecipe.
Run this after inference to train an MLP classifier on the embeddings.
"""

import os

from nvflare.app_common.np.recipes.fedavg import (  # we use the Numpy version of FedAvgRecipe here as the training is implemented with sklearn
    NumpyFedAvgRecipe,
)
from nvflare.recipe import SimEnv
from nvflare.recipe.utils import add_experiment_tracking

# Configuration
n_clients = 3
data_root = "/tmp/data/mixed_soft"
results_path = "/tmp/data/mixed_soft/results"
split_alpha = 1.0
embedding_dimensions = 320  # embedding dimensions of ESM2-8m

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
script_args = f"--data-root {data_root} --results-path {results_path} --aggregation-epochs 4 --lr 1e-5 --batch-size 128 --embedding-dimensions {embedding_dimensions}"

# Create FedAvgRecipe for MLP training
recipe = NumpyFedAvgRecipe(
    name=job_name, min_clients=n_clients, num_rounds=100, train_script="client.py", train_args=script_args
)

# Enable TensorBoard tracking
add_experiment_tracking(recipe, "tensorboard")

# Run simulation
env = SimEnv(
    num_clients=n_clients,
    workspace_root=f"/tmp/nvflare/bionemo/{job_name}_alpha{split_alpha}",
    num_threads=n_clients,  # MLP is lightweight, can run in parallel
)
run = recipe.execute(env)
print()
print("Job Status:", run.get_status())
print("Results:", run.get_result())
print()
print("\nView TensorBoard results with:")
print(f"tensorboard --logdir /tmp/nvflare/bionemo/{job_name}_alpha{split_alpha}")
