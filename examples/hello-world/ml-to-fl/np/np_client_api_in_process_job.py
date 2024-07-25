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

from nvflare import FedAvg, FedJob
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.client.config import ExchangeFormat


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="code/train_full.py")
    parser.add_argument("--params_transfer_type", type=str, default="FULL")

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    train_script = args.script
    params_transfer_type = args.params_transfer_type

    job = FedJob(name="np_client_api")

    # Define the controller workflow and send to server
    controller = FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=job.as_id(NPModelPersistor()))
    job.to(controller, "server")

    # Add MLflow Receiver for metrics streaming
    if train_script == "train_metrics.py":
        receiver = MLflowReceiver(
            tracking_uri="file:///tmp/nvflare/jobs/workdir/server/simulate_job/mlruns",
            kwargs={
                "experiment_name": "nvflare-fedavg-np-experiment",
                "run_name": "nvflare-fedavg-np-with-mlflow",
                "experiment_tags": {"mlflow.note.content": "## **NVFlare FedAvg Numpy experiment with MLflow**"},
                "run_tags": {"mlflow.note.content": "## Federated Experiment tracking with MLflow.\n"},
            },
            artifact_location="artifacts",
            events=["fed.analytix_log_stats"],
        )
        job.to(receiver, "server")

    # Add clients
    for i in range(n_clients):
        executor = InProcessClientAPIExecutor(
            task_script_path=train_script,
            params_exchange_format=ExchangeFormat.NUMPY,
            params_transfer_type=params_transfer_type,
        )

        job.to(executor, f"site-{i}", gpu=0)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")


if __name__ == "__main__":
    main()
