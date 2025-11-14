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

from nvflare import FilterType
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_common.widgets.decomposer_reg import DecomposerRegister
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import BaseScriptRunner

sys.path.append(os.path.join(os.getcwd(), ".."))  # include parent folder in path
from bionemo_filters import BioNeMoParamsFilter, BioNeMoStateDictFilter


def main(args):
    # Create BaseFedJob with initial model
    job = BaseFedJob(name=f"{args.exp_name}_sabdab_esm2_{args.model}")

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
    )
    job.to_server(controller)
    job.to_server(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]))

    checkpoint_path = load(f"esm2/{args.model}:2.0")
    print(f"Downloaded {args.model} to {checkpoint_path}")

    # Define unique strings describing the classes for classification so we can use the same label vocabulary on each client.
    classes = "pos,neg"

    # Add clients
    for i in range(args.num_clients):
        client_name = f"site-{i + 1}"

        # define data paths
        # We use the same validation set for each client to make their metrics comparable
        val_data_path = "/tmp/data/sabdab_chen/val/sabdab_chen_valid.csv"
        if "central" in args.exp_name:
            print("Simulating central training...")
            assert args.num_clients == 1, "Use num_clients=1 for simulating 'central' training setting."
            assert args.num_rounds == 1, "Use num_rounds=1 for simulating 'central' training setting."
            train_data_path = "/tmp/data/sabdab_chen/train/sabdab_chen_full_train.csv"
            val_check_interval = int(args.local_steps / 20)  # 20 times per training
        else:  # local or fedavg setting
            train_data_path = f"/tmp/data/sabdab_chen/train/sabdab_chen_{client_name}_train.csv"
            if args.num_rounds > 1:
                val_check_interval = args.local_steps
            else:
                val_check_interval = int(args.local_steps / 20)  # 20 times per training

        # define training script arguments
        # precision = "bf16-mixed"
        precision = "fp32"
        script_args = f"--restore-from-checkpoint-path {checkpoint_path} --train-data-path {train_data_path} --valid-data-path {val_data_path} --config-class ESM2FineTuneSeqConfig --dataset-class InMemorySingleValueDataset --task-type classification --mlp-ft-dropout 0.1 --mlp-hidden-size 256 --mlp-target-size 2 --experiment-name {job.name} --num-steps {args.local_steps} --num-gpus 1 --val-check-interval {val_check_interval} --log-every-n-steps 10 --lr 1e-4 --lr-multiplier 5 --scale-lr-layer classification_head --result-dir bionemo --micro-batch-size 64 --precision {precision} --save-top-k 1 --limit-val-batches 1.0 --classes {classes}"
        print(f"Running {args.train_script} with args: {script_args}")

        # Define training script runner
        runner = BaseScriptRunner(
            script=args.train_script,
            launch_external_process=True,
            framework="pytorch",
            server_expected_format="pytorch",
            # bionemo script is launched new at every FL round. Adds a shutdown grace period to make sure bionemo can save the local model
            launcher=SubprocessLauncher(
                script=f"python3 custom/{args.train_script} {script_args}", launch_once=False, shutdown_timeout=100.0
            ),
        )
        job.to(runner, client_name)
        job.to(
            BioNeMoParamsFilter(precision), client_name, tasks=["train", "validate"], filter_type=FilterType.TASK_DATA
        )
        job.to(BioNeMoStateDictFilter(), client_name, tasks=["train", "validate"], filter_type=FilterType.TASK_RESULT)
        job.to(DecomposerRegister(["nvflare.app_opt.pt.decomposers.TensorDecomposer"]), client_name)

    job.export_job("./exported_jobs")
    job.simulator_run(f"/tmp/nvflare/bionemo/sabdab/{job.name}", gpu=args.sim_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, help="Number of clients", required=False, default=1)
    parser.add_argument("--num_rounds", type=int, help="Number of rounds", required=False, default=30)
    parser.add_argument("--local_steps", type=int, help="Number of rounds", required=False, default=10)
    parser.add_argument(
        "--train_script", type=str, help="Training script", required=False, default="../finetune_esm2.py"
    )
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
