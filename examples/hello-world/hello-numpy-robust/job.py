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

"""Hello-world NumPy example with optional robust median aggregation."""

import argparse

from custom_aggregators import MedianAggregator

from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv, add_experiment_tracking

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--update_type", type=str, default="full", choices=["full", "diff"])
    parser.add_argument("--aggregator", type=str, default="default", choices=["default", "median"])
    parser.add_argument(
        "--poison_client_name",
        type=str,
        default="site-1",
        help="Client name to poison (for simulation only). Use empty string to disable poisoning.",
    )
    parser.add_argument("--poison_scale", type=float, default=1000.0)
    parser.add_argument("--tracking", type=str, default="none", choices=["none", "tensorboard"])
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--log_config",
        type=str,
        default=None,
        help="Log config mode ('concise', 'full', 'verbose'), filepath to a log config json file, or level.",
    )
    return parser.parse_args()


def _get_aggregator(aggregator_name: str):
    if aggregator_name == "median":
        print("Using MedianAggregator for robust aggregation")
        return MedianAggregator()
    print("Using default FedAvg aggregator")
    return None


def main():
    args = define_parser()

    train_args = (
        f"--update_type {args.update_type} "
        f"--poison_client_name {args.poison_client_name} "
        f"--poison_scale {args.poison_scale}"
    )

    recipe = NumpyFedAvgRecipe(
        name="hello-numpy-robust",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
        train_args=train_args,
        launch_external_process=args.launch_process,
        params_transfer_type=TransferType.FULL if args.update_type == "full" else TransferType.DIFF,
        aggregator=_get_aggregator(args.aggregator),
    )

    if args.tracking == "tensorboard":
        add_experiment_tracking(recipe, tracking_type="tensorboard")

    if args.export_config:
        job_dir = "/tmp/nvflare/jobs/job_config"
        recipe.export(job_dir)
        print(f"Job config exported to {job_dir}")
        return

    env = SimEnv(num_clients=args.n_clients, log_config=args.log_config)
    run = recipe.execute(env)
    print()
    print("Result can be found in :", run.get_result())
    print("Job Status is:", run.get_status())
    print()


if __name__ == "__main__":
    main()
