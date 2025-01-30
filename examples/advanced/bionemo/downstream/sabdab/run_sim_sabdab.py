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

from nvflare.job_config.script_runner import BaseScriptRunner

# Choose from one of the available jobs
job_name = "central_sabdab_esm1nv"
n_clients = 1
# job_name = "local_sabdab_esm1nv"; n_clients = 6
# job_name = "fedavg_sabdab_esm1nv"; n_clients = 6

#simulator = SimulatorRunner(
#    job_folder=f"jobs/{job_name}", workspace=f"/tmp/nvflare/results/{job_name}", n_clients=n_clients, threads=n_clients
#)
#run_status = simulator.run()
#print("Simulator finished with run_status", run_status)


import argparse
import logging

from nvflare import FilterType
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher


class ReceiveFilter(DXOFilter):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)
        
    def process_dxo(self, dxo, shareable, fl_ctx):    
        #self.logger.info(f"#######RECEIVING DXO: {dxo}")
        print(f"#######RECEIVING DXO: {dxo}")
        return dxo


class SendFilter(DXOFilter):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)
        
    def process_dxo(self, dxo, shareable, fl_ctx):    
        #self.logger.info(f"#######SENDING DXO: {dxo}")
        print(f"#######SENDING DXO: {dxo}")
        return dxo


def main(n_clients, num_rounds, train_script):
    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="central_sabdab_esm2nv"
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)
    #job.to_server(PTFileModelPersistor(), id="persistor")

    checkpoint_path = "/root/.cache/bionemo/2957b2c36d5978d0f595d6f1b72104b312621cf0329209086537b613c1c96d16-esm2_hf_converted_8m_checkpoint.tar.gz.untar"
    
    train_data_path = "/tmp/data/sabdab_chen/train/sabdab_chen_full_train.csv"
    val_data_path = "/tmp/data/sabdab_chen/val/sabdab_chen_valid.csv"
    
    # Add clients
    for i in range(n_clients):
        client_name = f"site-{i+1}"
        runner = BaseScriptRunner(script=train_script,
                             launch_external_process=True,
                             framework="pytorch",
                             params_exchange_format="pytorch",
                             launcher=SubprocessLauncher(script=f"python custom/{train_script} --restore-from-checkpoint-path {checkpoint_path} --train-data-path {train_data_path} --valid-data-path {val_data_path} --config-class ESM2FineTuneSeqConfig --dataset-class InMemorySingleValueDataset --experiment-name {job.name} --num-steps 10 --num-gpus 1 --val-check-interval 10 --log-every-n-steps 10 --lr 5e-3 --lr-multiplier 1e2 --scale-lr-layer regression_head --result-dir . --micro-batch-size 2 --num-gpus 1 --precision bf16-mixed", launch_once=False))
        job.to(runner, client_name)
        job.to(ReceiveFilter(), client_name, tasks=["*"], filter_type=FilterType.TASK_DATA)
        job.to(SendFilter(), client_name, tasks=["*"], filter_type=FilterType.TASK_RESULT)

    job.export_job("./exported_jobs")
    job.simulator_run(f"/tmp/nvflare/results/{job.name}", gpu="0")  # run all clients on the same GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, help="Number of clients", required=False, default=1)
    parser.add_argument("--num_rounds", type=str, help="Number of rounds", required=False, default=30)
    parser.add_argument("--train_script", type=str, help="Training script", required=False, default="finetune_esm2.py")  # TODO: "finetune_esm2.py"

    args = parser.parse_args()    
    
    main(args.n_clients, args.num_rounds, args.train_script)
    