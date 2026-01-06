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
Job configuration for PyTorch Lightning DDP federated learning.
"""

import argparse

from model import LitNet

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--use_tracking", action="store_true", help="Enable TensorBoard tracking")
    parser.add_argument("--export_config", action="store_true", help="Export job config only")
    return parser.parse_args()


def main():
    args = define_parser()

    train_script = "client.py"
    initial_model = LitNet()

    # Build train_args based on mode
    train_args = ""
    if args.use_tracking:
        train_args = "--use_tracking"

    # DDP modes require external process launch
    launch_external = True

    command = "python3 -u"

    recipe = FedAvgRecipe(
        name="lightning_ddp",
        initial_model=initial_model,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script=train_script,
        train_args=train_args,
        launch_external_process=launch_external,
        command=command,
    )

    if args.use_tracking:
        add_experiment_tracking(recipe, tracking_type="tensorboard")

    if args.export_config:
        recipe.export("/tmp/nvflare/jobs/job_config")
        print("Job config exported to /tmp/nvflare/jobs/job_config")
    else:
        env = SimEnv(num_clients=args.n_clients)
        run = recipe.execute(env)
        print()
        print("Result:", run.get_result())
        print("Status:", run.get_status())
        print()


if __name__ == "__main__":
    main()
