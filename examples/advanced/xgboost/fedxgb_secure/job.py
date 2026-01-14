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

import argparse

from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
from nvflare.app_opt.xgboost.recipes import XGBHistogramRecipe, XGBVerticalRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser(description="Secure Federated XGBoost with Homomorphic Encryption")
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=3, help="Total number of training rounds")
    parser.add_argument("--nthread", type=int, default=1, help="nthread for xgboost")
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist for best perf"
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="dataset split mode, horizontal or vertical",
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Whether to use secure training with Homomorphic Encryption",
    )
    return parser.parse_args()


def _get_job_name(args) -> str:
    if args.secure:
        return f"{args.data_split_mode}_secure"
    else:
        return f"{args.data_split_mode}"


def main():
    args = define_parser()
    job_name = _get_job_name(args)
    dataset_path = args.data_root

    # XGBoost parameters
    xgb_params = {
        "max_depth": 3,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": args.tree_method,
        "nthread": args.nthread,
    }

    # Create recipe based on data split mode
    if args.data_split_mode == "horizontal":
        # Horizontal histogram-based XGBoost
        recipe = XGBHistogramRecipe(
            name=job_name,
            min_clients=args.site_num,
            num_rounds=args.round_num,
            algorithm="histogram_v2",
            early_stopping_rounds=3,
            use_gpus=False,
            secure=args.secure,
            xgb_params=xgb_params,
        )

        # Add data loaders to each client
        # Note: CSVDataLoader automatically handles client-specific data loading.
        # Even though we pass the same folder path, each client will load its own data
        # from {dataset_path}/{client_id}/ at runtime (e.g., site-1 loads from site-1/ subdirectory)
        for i in range(1, args.site_num + 1):
            dataloader = CSVDataLoader(folder=dataset_path)
            recipe.add_to_client(f"site-{i}", dataloader)

    else:  # vertical
        # Vertical histogram-based XGBoost
        # Generate client ranks for secure training
        client_ranks = {f"site-{i}": i - 1 for i in range(1, args.site_num + 1)}

        recipe = XGBVerticalRecipe(
            name=job_name,
            min_clients=args.site_num,
            num_rounds=args.round_num,
            label_owner="site-1",  # First site owns labels
            early_stopping_rounds=3,
            use_gpus=False,
            secure=args.secure,
            client_ranks=client_ranks if args.secure else None,
            xgb_params=xgb_params,
        )

        # Add data loaders to each client
        # Note: CSVDataLoader automatically handles client-specific data loading.
        # For vertical mode, each client loads different feature columns from its subdirectory:
        # - site-1 (rank 0): loads features + labels from {dataset_path}/site-1/
        # - site-2, site-3: load different features (no labels) from their respective subdirectories
        for i in range(1, args.site_num + 1):
            dataloader = CSVDataLoader(folder=dataset_path)
            recipe.add_to_client(f"site-{i}", dataloader)

    # Export and run
    env = SimEnv(num_clients=args.site_num)
    run = recipe.execute(env)
    run.export_job(f"/tmp/nvflare/workspace/fedxgb_secure/train_fl/jobs/{job_name}")

    # Run the job except for secure horizontal (which needs special context setup)
    if args.data_split_mode == "horizontal" and args.secure:
        print(
            "Secure horizontal job prepared, to run it with simulator, tenseal context needs to be added, "
            "please see README for next steps."
        )
    else:
        run.simulator_run(f"/tmp/nvflare/workspace/fedxgb_secure/train_fl/works/{job_name}")


if __name__ == "__main__":
    main()
