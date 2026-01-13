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

from local_psi import LocalPSI
from vertical_data_loader import VerticalDataLoader

from nvflare.app_common.psi.dh_psi.dh_psi_controller import DhPSIController
from nvflare.app_common.psi.file_psi_writer import FilePSIWriter
from nvflare.app_common.psi.psi_executor import PSIExecutor
from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import DhPSITaskHandler
from nvflare.app_opt.xgboost.recipes import XGBVerticalRecipe
from nvflare.job_config.api import FedJob
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_path",
        type=str,
        default="/tmp/nvflare/dataset/xgboost_higgs_vertical/{SITE_NAME}/higgs.data.csv",
        help="Path to data split files for each site (use {SITE_NAME} placeholder)",
    )
    parser.add_argument(
        "--psi_output_path",
        type=str,
        default="psi/intersection.txt",
        help="PSI output path (relative to client workspace)",
    )
    parser.add_argument("--site_num", type=int, default=2, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")
    parser.add_argument("--id_col", type=str, default="uid", help="Column name for sample IDs")
    parser.add_argument(
        "--label_owner", type=str, default="site-1", help="Client ID that owns the labels (e.g., 'site-1')"
    )
    parser.add_argument("--train_proportion", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--max_depth", type=int, default=8, help="max_depth for xgboost")
    parser.add_argument("--eta", type=float, default=0.1, help="learning rate (eta) for xgboost")
    parser.add_argument("--objective", type=str, default="binary:logistic", help="objective for xgboost")
    parser.add_argument("--eval_metric", type=str, default="auc", help="eval_metric for xgboost")
    parser.add_argument("--early_stopping_rounds", type=int, default=3, help="early stopping rounds")
    parser.add_argument("--nthread", type=int, default=16, help="nthread for xgboost")
    parser.add_argument("--tree_method", type=str, default="hist", help="tree_method for xgboost")
    parser.add_argument("--run_psi", action="store_true", help="Run PSI job (required before training)")
    parser.add_argument("--run_training", action="store_true", help="Run training job (requires PSI results)")

    return parser.parse_args()


def run_psi_job(args):
    """Run Private Set Intersection (PSI) job to compute sample intersection."""
    print("\n" + "=" * 80)
    print("STEP 1: Running PSI Job to compute sample intersection")
    print("=" * 80 + "\n")

    job_name = "xgboost_vertical_psi"
    job = FedJob(name=job_name, min_clients=args.site_num)

    # PSI Controller
    controller = DhPSIController()
    job.to_server(controller)

    # PSI Executor
    executor = PSIExecutor(psi_algo_id="dh_psi")
    job.to_clients(executor, id="psi_executor", tasks=["PSI"])

    # Local PSI handler
    local_psi = LocalPSI(psi_writer_id="psi_writer", data_split_path=args.data_split_path, id_col=args.id_col)
    job.to_clients(local_psi, id="local_psi")

    # PSI task handler
    task_handler = DhPSITaskHandler(local_psi_id="local_psi")
    job.to_clients(task_handler, id="dh_psi")

    # PSI writer
    psi_writer = FilePSIWriter(output_path=args.psi_output_path)
    job.to_clients(psi_writer, id="psi_writer")

    # Run PSI job
    env = SimEnv()
    env.run(job, work_dir=f"/tmp/nvflare/workspace/works/{job_name}")

    print("\n" + "=" * 80)
    print(f"PSI Complete! Intersection files saved to: {args.psi_output_path}")
    print("=" * 80 + "\n")


def run_training_job(args):
    """Run vertical XGBoost training job (requires PSI results)."""
    print("\n" + "=" * 80)
    print("STEP 2: Running Vertical XGBoost Training")
    print("=" * 80 + "\n")

    # Construct PSI path based on PSI job output location
    psi_path = f"/tmp/nvflare/workspace/works/xgboost_vertical_psi/{{SITE_NAME}}/simulate_job/{{SITE_NAME}}/{args.psi_output_path}"

    # XGBoost parameters
    xgb_params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "tree_method": args.tree_method,
        "nthread": args.nthread,
    }

    # Create vertical XGBoost recipe
    recipe = XGBVerticalRecipe(
        name="xgboost_vertical",
        min_clients=args.site_num,
        num_rounds=args.round_num,
        label_owner=args.label_owner,
        early_stopping_rounds=args.early_stopping_rounds,
        xgb_params=xgb_params,
    )

    # Add data loaders to each client
    for site_id in range(1, args.site_num + 1):
        site_name = f"site-{site_id}"
        data_loader = VerticalDataLoader(
            data_split_path=args.data_split_path.replace("{SITE_NAME}", site_name),
            psi_path=psi_path.replace("{SITE_NAME}", site_name),
            id_col=args.id_col,
            label_owner=args.label_owner,
            train_proportion=args.train_proportion,
        )
        recipe.add_to_client(site_name, data_loader)

    # Run training
    env = SimEnv()
    env.run(recipe, work_dir="/tmp/nvflare/workspace/works/xgboost_vertical")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80 + "\n")


def main():
    args = define_parser()

    # Validate arguments
    if not args.run_psi and not args.run_training:
        print("Error: Must specify --run_psi and/or --run_training")
        print("\nUsage examples:")
        print("  1. Run PSI only:")
        print("     python job_vertical.py --run_psi")
        print("\n  2. Run training only (requires PSI results):")
        print("     python job_vertical.py --run_training")
        print("\n  3. Run both (PSI then training):")
        print("     python job_vertical.py --run_psi --run_training")
        return

    # Run PSI if requested
    if args.run_psi:
        run_psi_job(args)

    # Run training if requested
    if args.run_training:
        run_training_job(args)


if __name__ == "__main__":
    main()
