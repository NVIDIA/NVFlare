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

from higgs_data_loader import HIGGSDataLoader

from nvflare.app_opt.xgboost.recipes import XGBHistogramRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/xgboost_higgs",
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=2, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")
    parser.add_argument(
        "--training_algo",
        type=str,
        default="histogram_v2",
        choices=["histogram", "histogram_v2"],
        help="Training algorithm (histogram or histogram_v2)",
    )
    parser.add_argument("--split_method", type=str, default="uniform", help="How to split the dataset")
    parser.add_argument("--nthread", type=int, default=16, help="nthread for xgboost")
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist for best perf"
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="horizontal",
        choices=["horizontal"],
        help="dataset split mode (only horizontal supported by this recipe)",
    )
    parser.add_argument("--max_depth", type=int, default=8, help="max_depth for xgboost")
    parser.add_argument("--eta", type=float, default=0.1, help="learning rate (eta) for xgboost")
    parser.add_argument("--objective", type=str, default="binary:logistic", help="objective for xgboost")
    parser.add_argument("--eval_metric", type=str, default="auc", help="eval_metric for xgboost")
    parser.add_argument("--early_stopping_rounds", type=int, default=2, help="early stopping rounds")
    parser.add_argument("--use_gpus", action="store_true", help="use GPUs for training")

    return parser.parse_args()


def _get_job_name(args) -> str:
    return f"higgs_{args.site_num}_{args.training_algo}_{args.split_method}_split"


def _get_data_path(args) -> str:
    return f"{args.data_root}_{args.data_split_mode}/{args.site_num}_{args.split_method}"


def main():
    args = define_parser()
    job_name = _get_job_name(args)
    dataset_path = _get_data_path(args)

    # XGBoost parameters
    xgb_params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "tree_method": args.tree_method,
        "nthread": args.nthread,
    }

    # Create recipe
    recipe = XGBHistogramRecipe(
        name=job_name,
        min_clients=args.site_num,
        num_rounds=args.round_num,
        algorithm=args.training_algo,
        early_stopping_rounds=args.early_stopping_rounds,
        use_gpus=args.use_gpus,
        xgb_params=xgb_params,
    )

    # Add data loaders to each client
    for site_id in range(1, args.site_num + 1):
        data_loader = HIGGSDataLoader(data_split_filename=f"{dataset_path}/data_site-{site_id}.json")
        recipe.add_to_client(f"site-{site_id}", data_loader)

    # Run simulation
    env = SimEnv()
    env.run(recipe, work_dir=f"/tmp/nvflare/workspace/works/{job_name}")


if __name__ == "__main__":
    main()
