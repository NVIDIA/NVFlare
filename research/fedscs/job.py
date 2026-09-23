# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""FedSCS CIFAR-10 job."""

import argparse
import os

from src.fedavg_clipped_aggregator import FedAvgClippedAggregator
from src.fedscs_aggregator import FedSCSAggregator
from src.model import SimpleCNN

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv


def parse_args():
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="Run FedAvg or FedSCS on CIFAR-10.")
    parser.add_argument(
        "--method",
        choices=["fedavg", "fedavg_clipped", "fedscs"],
        default="fedscs",
        help="Aggregation method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
        help="Experiment/training seed.",
    )
    parser.add_argument(
        "--initial_ckpt",
        type=str,
        default=None,
        help="Absolute path to the initial model checkpoint.",
    )
    return parser.parse_args()


def main():
    """Create and execute the federated learning job."""
    args = parse_args()

    job_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(job_dir, "data")

    model = SimpleCNN()

    expected_schema = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    expected_dtypes = {name: str(value.dtype).replace("torch.", "") for name, value in model.state_dict().items()}

    aggregator = None

    if args.method == "fedavg_clipped":
        aggregator = FedAvgClippedAggregator(
            max_update_norm=10.0,
        )
    elif args.method == "fedscs":
        max_update_norm = 10.0

        aggregator = FedSCSAggregator(
            expected_schema=expected_schema,
            expected_dtypes=expected_dtypes,
            max_update_norm=max_update_norm,
        )

    train_args = f"--data_dir {data_dir} --seed {args.seed}"

    recipe = FedAvgRecipe(
        name=f"{args.method}_seed_{args.seed}",
        min_clients=5,
        num_rounds=10,
        model=model,
        initial_ckpt=args.initial_ckpt,
        train_script=os.path.join(job_dir, "client.py"),
        train_args=train_args,
        aggregator=aggregator,
        params_transfer_type=TransferType.DIFF,
        key_metric="global_accuracy",
    )

    recipe.execute(SimEnv(num_clients=5))


if __name__ == "__main__":
    main()
