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

from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/xgboost_higgs",
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
    parser.add_argument(
        "--round_num",
        type=int,
        default=None,
        help="Total number of training rounds (default: 1 for bagging, 100 for cyclic)",
    )
    parser.add_argument(
        "--training_algo",
        type=str,
        default="bagging",
        choices=["bagging", "cyclic"],
        help="Training algorithm (bagging or cyclic)",
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
    parser.add_argument("--use_gpus", action="store_true", help="use GPUs for training")
    parser.add_argument(
        "--lr_mode",
        type=str,
        default="uniform",
        choices=["uniform", "scaled"],
        help="learning rate mode (uniform or scaled)",
    )
    parser.add_argument("--num_local_parallel_tree", type=int, default=5, help="number of parallel trees per client")
    parser.add_argument("--local_subsample", type=float, default=0.8, help="subsample ratio for local training")

    return parser.parse_args()


def _get_job_name(args) -> str:
    return f"higgs_{args.site_num}_{args.training_algo}_{args.split_method}_split_{args.lr_mode}"


def _get_data_path(args) -> str:
    return f"{args.data_root}_{args.data_split_mode}/{args.site_num}_{args.split_method}"


def main():
    args = define_parser()
    job_name = _get_job_name(args)
    dataset_path = _get_data_path(args)

    # Build per-site configuration with data loaders
    per_site_config = {}
    for site_id in range(1, args.site_num + 1):
        site_name = f"site-{site_id}"
        data_loader = HIGGSDataLoader(data_split_filename=f"{dataset_path}/data_site-{site_id}.json")

        site_config = {"data_loader": data_loader}

        # For scaled lr_mode, add custom learning rate scale based on data size
        if args.lr_mode == "scaled":
            # In a real scenario, you'd calculate lr_scale based on actual data size
            # For now, we use a simple formula based on site_num
            lr_scale = 1.0 / args.site_num
            site_config["lr_scale"] = lr_scale

        per_site_config[site_name] = site_config

    # Create recipe
    recipe = XGBBaggingRecipe(
        name=job_name,
        min_clients=args.site_num,
        training_mode=args.training_algo,
        num_rounds=args.round_num,  # Will default based on training_mode if None
        num_local_parallel_tree=args.num_local_parallel_tree,
        local_subsample=args.local_subsample,
        learning_rate=args.eta,
        objective=args.objective,
        max_depth=args.max_depth,
        eval_metric=args.eval_metric,
        tree_method=args.tree_method,
        use_gpus=args.use_gpus,
        nthread=args.nthread,
        lr_mode=args.lr_mode,
        per_site_config=per_site_config,
    )

    # Run simulation
    env = SimEnv(num_clients=args.site_num)
    run = recipe.execute(env)
    print()
    print("Job Status:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
