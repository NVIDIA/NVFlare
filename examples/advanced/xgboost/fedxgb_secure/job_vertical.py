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

"""Vertical Secure Federated XGBoost with Homomorphic Encryption (HE)

This example demonstrates vertical histogram-based XGBoost with optional
Homomorphic Encryption for privacy-preserving federated learning.

In vertical federated learning:
- Different sites have different features for overlapping samples
- site-1 is the label owner (has features + labels)
- Other sites have different feature sets (no labels)

Run with defaults:
    python job_vertical.py

Run with secure training:
    python job_vertical.py --secure

Run with custom parameters:
    python job_vertical.py --site_num 5 --round_num 10 --secure
"""

import argparse

from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
from nvflare.app_opt.xgboost.recipes import XGBVerticalRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser(description="Vertical Secure Federated XGBoost with Homomorphic Encryption")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/xgb_dataset/vertical_xgb_data",
        help="Path to vertical dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=3, help="Total number of training rounds")
    parser.add_argument("--nthread", type=int, default=1, help="nthread for xgboost")
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist for best perf"
    )
    parser.add_argument(
        "--label_owner",
        type=str,
        default="site-1",
        help="Site that owns the labels (default: site-1)",
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Whether to use secure training with Homomorphic Encryption",
    )
    return parser.parse_args()


def main():
    args = define_parser()

    # Job name
    job_name = "vertical_secure" if args.secure else "vertical"

    # XGBoost parameters
    xgb_params = {
        "max_depth": 3,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": args.tree_method,
        "nthread": args.nthread,
    }

    # Generate client ranks if secure training is enabled
    # Maps client names to ranks (0-indexed)
    # Required for Homomorphic Encryption
    client_ranks = None
    if args.secure:
        client_ranks = {f"site-{i}": i - 1 for i in range(1, args.site_num + 1)}

    # Build per-site configuration with data loaders
    # Note: CSVDataLoader automatically handles client-specific data loading.
    # Each client loads from {data_root}/{client_id}/ at runtime
    # For vertical mode:
    # - Label owner (site-1): loads features + labels
    # - Other sites: load different features (no labels)
    per_site_config = {
        f"site-{i}": {"data_loader": CSVDataLoader(folder=args.data_root)} for i in range(1, args.site_num + 1)
    }

    # Create vertical XGBoost recipe
    recipe = XGBVerticalRecipe(
        name=job_name,
        min_clients=args.site_num,
        num_rounds=args.round_num,
        label_owner=args.label_owner,
        early_stopping_rounds=3,
        secure=args.secure,
        client_ranks=client_ranks,
        xgb_params=xgb_params,
        per_site_config=per_site_config,
    )

    # Export and run
    env = SimEnv(num_clients=args.site_num)
    run = recipe.execute(env)
    run.export_job(f"/tmp/nvflare/workspace/fedxgb_secure/train_fl/jobs/{job_name}")
    run.simulator_run(f"/tmp/nvflare/workspace/fedxgb_secure/train_fl/works/{job_name}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
