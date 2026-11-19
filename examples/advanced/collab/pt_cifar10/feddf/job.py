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

"""Compare synchronous weight averaging and KD using identical CIFAR-10 clients."""

import argparse

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

from ..fedavg.client import FedAvgClient
from .distiller import Cifar10DistillerFactory
from .prepare_data import DEFAULT_SPLIT_ROOT
from .server import FedDFServer


def make_recipe(args):
    if args.num_clients < 1:
        raise ValueError("num_clients must be positive")
    server = FedDFServer(
        num_rounds=args.num_rounds,
        aggregation=args.aggregation,
        split_root=args.split_root,
        seed=args.seed,
        kd_epochs=args.kd_epochs,
        distiller_factory=Cifar10DistillerFactory(args.distiller, args.kd_lr, args.temperature),
    )
    # Wire the synchronous server to the existing published FedAvg client.
    variant = "fedavg" if args.aggregation == "fedavg" else f"feddf_{args.distiller}"
    return CollabRecipe(
        job_name=f"collab_cifar10_{variant}_proxy",
        server=server,
        client=FedAvgClient(split_root=args.split_root, seed=args.seed),
        min_clients=args.num_clients,
        max_call_threads_for_server=max(4, args.num_clients),
        max_call_threads_for_client=4,
        sync_task_timeout=600,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", choices=("fedavg", "feddf"), default="feddf")
    parser.add_argument("--distiller", choices=("native", "modelopt", "torchdistill"), default="native")
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--kd-epochs", type=int, default=1)
    parser.add_argument("--kd-lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab-feddf")
    args = parser.parse_args()
    recipe = make_recipe(args)
    run = recipe.execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_root))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
