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
import os
import sys

from bionemo.core.data.load import load
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

sys.path.append(os.path.join(os.getcwd(), ".."))  # include parent folder in path
from bionemo_filters import BioNeMoExcludeParamsFilter, BioNeMoParamsFilter, BioNeMoStateDictFilter


def main(args):
    checkpoint_path = load(f"esm2/{args.model}:2.0")
    print(f"Downloaded {args.model} to {checkpoint_path}")

    # Determine validation check interval based on experiment type
    if "central" in args.exp_name:
        print("Simulating central training...")
        # in central training, we allow several clients to train on the same data (each with a different end point), but we only run one round of training
        assert args.num_rounds == 1, "Use num_rounds=1 for simulating 'central' training setting."
        val_check_interval = int(args.local_steps / 20)  # 20 times per training
    else:  # local or fedavg setting
        if args.num_rounds > 1:
            val_check_interval = args.local_steps
        else:
            val_check_interval = int(args.local_steps / 20)  # 20 times per training

    # Build training script arguments (same for all clients, data paths and label_column resolved in client.py)
    precision = "fp32"
    script_args = f"--restore-from-checkpoint-path {checkpoint_path} --train-data-path /tmp/placeholder --valid-data-path /tmp/placeholder --config-class ESM2FineTuneSeqConfig --dataset-class InMemorySingleValueDataset --task-type regression --mlp-ft-dropout 0.1 --mlp-hidden-size 256 --mlp-target-size 1 --experiment-name tap_esm2_{args.model} --num-steps {args.local_steps} --num-gpus 1 --val-check-interval {val_check_interval} --log-every-n-steps 10 --lr 5e-4 --lr-multiplier 1e3 --scale-lr-layer regression_head --result-dir bionemo --micro-batch-size 8 --precision {precision} --save-top-k 1 --limit-val-batches 1.0 --label-column placeholder --dataset-name tap --exp-name {args.exp_name}"
    print(f"Running {args.train_script} with base args (data paths and label_column will be resolved per-client)")

    # Create FedAvgRecipe
    job_name = f"{args.exp_name}_tap_esm2_{args.model}"
    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        train_script=f"../{args.train_script}",
        train_args=script_args,
        launch_external_process=True,
        command="python3",
        launch_once=False,
        shutdown_timeout=30,
        server_expected_format="pytorch",  # this will send pytorch tensors directly between clients and server, bypassing the need for numpy conversion
    )

    # Add custom components using recipe's filter API
    recipe.add_client_input_filter(BioNeMoParamsFilter(precision), tasks=["train", "validate"])
    recipe.add_client_output_filter(BioNeMoStateDictFilter(), tasks=["train", "validate"])
    # Do not share the regression head with the server; each client will train their personal endpoint in this example
    recipe.add_client_output_filter(
        BioNeMoExcludeParamsFilter(exclude_vars="regression_head"), tasks=["train", "validate"]
    )

    # Add decomposer register to server and clients
    recipe.job.to_server(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]))
    recipe.job.to_clients(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]))

    # Run simulation
    env = SimEnv(
        num_clients=args.num_clients, workspace_root=f"/tmp/nvflare/bionemo/tap/{job_name}", gpu_config=args.sim_gpus
    )
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, help="Number of clients", required=False, default=1)
    parser.add_argument("--num_rounds", type=int, help="Number of rounds", required=False, default=30)
    parser.add_argument("--local_steps", type=int, help="Number of local training steps", required=False, default=10)
    parser.add_argument("--train_script", type=str, help="Training script", required=False, default="client.py")
    parser.add_argument("--exp_name", type=str, help="Job name prefix", required=False, default="fedavg")
    parser.add_argument("--model", choices=["8m", "650m", "3b"], help="ESM2 model", required=False, default="8m")
    parser.add_argument(
        "--sim_gpus",
        type=str,
        help="GPU indexes to simulate clients, e.g., '0,1,2,3' if you want to run 4 clients, each on a separate GPU. By default run all clients on the same GPU 0.",
        required=False,
        default="0",
    )

    args = parser.parse_args()

    main(args)
