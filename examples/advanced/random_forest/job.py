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
Recipe-based Federated Random Forest using XGBoost Tree-based Bagging.

This example demonstrates federated Random Forest training using NVFlare's recipe API.
Each client trains a local sub-forest on their data, and these are aggregated on the server.

Usage:
    python job.py --n_clients 5 --local_subsample 0.5 --data_split_path /tmp/nvflare/random_forest/HIGGS/data_splits/5_uniform
"""

import argparse

from jobs.random_forest_base.app.custom.higgs_data_loader import HIGGSDataLoader

from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    """Define command-line arguments."""
    parser = argparse.ArgumentParser(description="Federated Random Forest using XGBoost Bagging Recipe")
    parser.add_argument("--n_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of training rounds (default: 1 for RF)")
    parser.add_argument(
        "--num_local_parallel_tree", type=int, default=5, help="Number of parallel trees per client (default: 5)"
    )
    parser.add_argument(
        "--local_subsample", type=float, default=0.5, help="Subsample ratio for local training (default: 0.5)"
    )
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum tree depth (default: 8)")
    parser.add_argument(
        "--objective", type=str, default="binary:logistic", help="Learning objective (default: binary:logistic)"
    )
    parser.add_argument("--eval_metric", type=str, default="auc", help="Evaluation metric (default: auc)")
    parser.add_argument("--tree_method", type=str, default="hist", help="Tree construction method (default: hist)")
    parser.add_argument("--use_gpus", action="store_true", help="Use GPUs for training")
    parser.add_argument("--nthread", type=int, default=16, help="Number of threads (default: 16)")
    parser.add_argument(
        "--lr_mode", type=str, default="uniform", choices=["uniform", "scaled"], help="Learning rate mode"
    )
    parser.add_argument(
        "--data_split_path",
        type=str,
        default="/tmp/nvflare/random_forest/HIGGS/data_splits/5_uniform",
        help="Path to data split directory",
    )
    return parser.parse_args()


def main():
    """Main function to set up and run the federated Random Forest job."""
    args = define_parser()

    # Create the XGBoost Bagging Recipe
    recipe = XGBBaggingRecipe(
        name=f"random_forest_{args.n_clients}clients",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        num_client_bagging=args.n_clients,
        num_local_parallel_tree=args.num_local_parallel_tree,
        local_subsample=args.local_subsample,
        learning_rate=args.learning_rate,
        objective=args.objective,
        max_depth=args.max_depth,
        eval_metric=args.eval_metric,
        tree_method=args.tree_method,
        use_gpus=args.use_gpus,
        nthread=args.nthread,
        lr_mode=args.lr_mode,
        save_name="xgboost_model.json",
        data_loader_id="dataloader",
    )

    # Add executor and data loader to each client
    for site_id in range(1, args.n_clients + 1):
        data_split_file = f"{args.data_split_path}/data_site-{site_id}.json"
        dataloader = HIGGSDataLoader(data_split_filename=data_split_file)
        recipe.add_to_client(f"site-{site_id}", dataloader, lr_scale=1.0)

    # Add experiment tracking (TensorBoard)
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Execute the recipe in simulation environment
    # Create client list for SimEnv
    client_names = [f"site-{i}" for i in range(1, args.n_clients + 1)]
    env = SimEnv(clients=client_names, num_threads=args.n_clients)
    run = recipe.execute(env)

    # Print results
    print(f"Job Status is: {run.get_status()}")
    print(f"Result can be found in: {run.get_result()}")


if __name__ == "__main__":
    main()
