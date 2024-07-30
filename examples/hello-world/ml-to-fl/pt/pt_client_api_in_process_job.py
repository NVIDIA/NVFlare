# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from src.lit_net import LitNet
from src.net import Net

from nvflare import FedAvg, FedJob, ScriptExecutor
from nvflare.client.config import ExchangeFormat


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/cifar10_fl.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    train_script = args.script
    key_metric = args.key_metric

    job = FedJob(name="pt_client_api_in_process", key_metric=key_metric)

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    job.to(LitNet() if "lightning" in train_script else Net(), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptExecutor(
            task_script_path=train_script,
            task_script_args="",
            params_exchange_format=ExchangeFormat.PYTORCH,
            params_transfer_type="DIFF",
        )
        job.to(executor, f"site-{i+1}", gpu=0)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")


if __name__ == "__main__":
    main()
