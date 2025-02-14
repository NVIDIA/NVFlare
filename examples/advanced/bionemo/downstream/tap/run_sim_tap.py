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
import logging

from bionemo.core.data.load import load
from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher


def main(args):
    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name=f"{args.exp_name}_sabdab_esm2_{args.model}"
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
    )
    job.to_server(controller)
    #job.to_server(PTFileModelPersistor(), id="persistor")  # TODO: load ckpt

    checkpoint_path = load(f"esm2/{args.model}:2.0")
    print(f"Don {args.model} to {checkpoint_path}")
    
    # Add clients
    for i in range(args.num_clients):
        client_name = f"site-{i+1}"

        # define data paths
        # We use the same validation set for each client to make their metrics comparable
        val_data_path = "/tmp/data/tap/val/tap_valid.csv"
        if "central" in args.exp_name:
            print("Simulating central training...")
            check_interval = 10
            assert args.num_clients == 1, "Use num_clients=1 for simulating 'central' training setting."
            assert args.num_rounds == 1, "Use num_rounds=1 for simulating 'central' training setting."
            train_data_path = "/tmp/data/tap/train/tap_full_train.csv"
        else: # local or fedavg setting
            check_interval = int(args.local_steps/2)
            train_data_path = f"/tmp/data/tap/train/tap_{client_name}_train.csv"            
        
        # define training script arguments
        script_args = f"--restore-from-checkpoint-path {checkpoint_path} --train-data-path {train_data_path} --valid-data-path {val_data_path} --config-class ESM2FineTuneSeqConfig --dataset-class InMemorySingleValueDataset --task-type regression --mlp-ft-dropout 0.25 --mlp-hidden-size 128 --mlp-target-size 1 --experiment-name {job.name} --num-steps {args.local_steps} --num-gpus 1 --val-check-interval {check_interval} --log-every-n-steps {check_interval} --lr 1e-6 --lr-multiplier 5e2 --scale-lr-layer regression_head --result-dir .  --micro-batch-size 32 --precision fp32 --save-top-k 1"
        print(f"Running {args.train_script} with args: {script_args}")
        
        # Define training script runner
        runner = ScriptRunner(script=args.train_script,
                             script_args=script_args,
                             launch_external_process=True,
                             framework="pytorch",
                             params_exchange_format="pytorch")
        job.to(runner, client_name)

    job.export_job("./exported_jobs")
    job.simulator_run(f"/tmp/nvflare/results/{job.name}", gpu=args.sim_gpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, help="Number of clients", required=False, default=1)
    parser.add_argument("--num_rounds", type=int, help="Number of rounds", required=False, default=30)
    parser.add_argument("--local_steps", type=int, help="Number of rounds", required=False, default=10)
    parser.add_argument("--train_script", type=str, help="Training script", required=False, default="../finetune_esm2.py")
    parser.add_argument("--exp_name", type=str, help="Job name prefix", required=False, default="fedavg")
    parser.add_argument("--model", choices=["8m", "650m", "3b"], help="ESM2 model", required=False, default="8m")
    parser.add_argument("--sim_gpus", type=str, help="GPU indexes to simulate clients, e.g., '0,1,2,3' if you want to run 4 clients, each on a separate GPU. By default run all clients on the same GPU 0.", required=False, default="0")

    args = parser.parse_args()    
    
    main(args)
    