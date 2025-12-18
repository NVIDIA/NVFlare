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
This code shows how to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm
and run it under different environments
"""
import argparse

from nvflare.apis.dxo import DataKind
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--update_type", type=str, default="full", choices=["full", "diff"])
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    launch_process = args.launch_process

    train_args = f"--update_type {args.update_type}"
    recipe = NumpyFedAvgRecipe(
        name="hello-numpy",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
        train_args=train_args,
        launch_external_process=launch_process,
        aggregator_data_kind=DataKind.WEIGHTS if args.update_type == "full" else DataKind.WEIGHT_DIFF,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.export_config:
        job_dir = "/tmp/nvflare/jobs/job_config"
        recipe.export(job_dir)
        print(f"Job config exported to {job_dir}")
    else:
        env = SimEnv(num_clients=n_clients)
        run = recipe.execute(env)
        print()
        print("Result can be found in :", run.get_result())
        print("Job Status is:", run.get_status())
        print()


if __name__ == "__main__":
    main()
