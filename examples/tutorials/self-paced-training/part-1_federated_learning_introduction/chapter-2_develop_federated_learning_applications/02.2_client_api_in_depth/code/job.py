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

import argparse

from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--execution_mode", choices=["in_process", "external_process"], default="in_process")
    parser.add_argument("--transfer_type", choices=["full", "diff"], default="full")
    parser.add_argument("--workspace_dir", default="/tmp/nvflare/chapter2/client_api")
    args = parser.parse_args()
    if args.num_rounds < 1 or args.local_steps < 1 or args.learning_rate <= 0:
        parser.error("num_rounds, local_steps, and learning_rate must be positive")

    recipe = NumpyFedAvgRecipe(
        name=f"client_api_{args.execution_mode}_{args.transfer_type}",
        model=[0.0, 0.0, 0.0],
        min_clients=2,
        num_rounds=args.num_rounds,
        train_script="client.py",
        train_args=f"--local_steps {args.local_steps} --learning_rate {args.learning_rate}",
        launch_external_process=args.execution_mode == "external_process",
        command="python -u",
        params_transfer_type=TransferType.FULL if args.transfer_type == "full" else TransferType.DIFF,
        key_metric="mse",
        key_metric_mode="min",
    )
    run = recipe.execute(SimEnv(num_clients=2, workspace_root=args.workspace_dir, log_config="progress"))
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
