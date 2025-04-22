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
Run FL for AMPLIFY sequence regression.
"""

import argparse
import os

from src.filters import ExcludeParamsFilter
from src.model import AmplifyRegressor

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

tasks = ["aggregation", "binding", "expression", "immunogenicity", "polyreactivity", "tm"]


def main(args):
    train_script = "src/finetune_seqclassification_fl.py"

    # Parse layer sizes from string to list of integers
    layer_sizes = [int(size) for size in args.layer_sizes.split(",")]

    # Define the initial global model on the server
    model = AmplifyRegressor(pretrained_model_name_or_path=args.pretrained_model, layer_sizes=layer_sizes)

    # Create BaseFedJob with initial model
    job = BaseFedJob(name=f"amplify_seqregression_{args.exp_name}", initial_model=model)

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=len(tasks),
        num_rounds=args.num_rounds,
    )
    job.to_server(controller)

    # Add clients using "task" as client_name
    for client_name in tasks:
        train_csv = os.path.join(args.data_root, client_name, "train_data.csv")
        test_csv = os.path.join(args.data_root, client_name, "test_data.csv")
        runner = ScriptRunner(
            script=train_script,
            script_args=f"--n_epochs {args.local_epochs} --pretrained_model {args.pretrained_model} --layer_sizes {args.layer_sizes} --train_csv {train_csv} --test_csv {test_csv}",
        )
        job.to(runner, client_name)

        job.to(
            ExcludeParamsFilter(exclude_vars="regressor"),
            client_name,
            tasks=["train", "validate"],
            filter_type=FilterType.TASK_RESULT,
        )  # do not share the regression head with the server; each client will train their personal endpoint in this example

    job.export_job("./job_configs")  # optionally save the job configs (not needed for simulation)
    job.simulator_run(f"/tmp/nvflare/AMPLIFY/multitask/{job.name}", gpu=args.sim_gpus, log_config="full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, help="Number of rounds", required=False, default=30)
    parser.add_argument("--local_epochs", type=int, help="Number of local epochs", required=False, default=10)
    parser.add_argument("--exp_name", type=str, help="Job name prefix", required=False, default="fedavg")
    parser.add_argument(
        "--pretrained_model",
        choices=["chandar-lab/AMPLIFY_120M", "chandar-lab/AMPLIFY_350M"],
        help="AMPLIFY model",
        required=False,
        default="chandar-lab/AMPLIFY_120M",
    )
    parser.add_argument(
        "--sim_gpus",
        type=str,
        help="GPU indexes to simulate clients, e.g., '0,1,2,3' if you want to run 4 clients, each on a separate GPU. By default run all clients on the same GPU 0.",
        required=False,
        default="0",
    )
    parser.add_argument(
        "--layer_sizes",
        type=str,
        default="128,64,32",
        help="Comma-separated list of layer sizes for the regression MLP",
    )
    parser.add_argument(
        "--data_root", type=str, default=f"{os.getcwd()}/FLAb/data_fl", help="Root directory for training and test data"
    )
    args = parser.parse_args()

    main(args)
