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

from nvflare import FedJob
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/train_full.py")
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    launch_process = args.launch_process
    export_config = args.export_config

    job = FedJob(name="np_client_api")

    persistor_id = job.to_server(NPModelPersistor(), "persistor")

    # Define the controller workflow and send to server
    controller = FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=persistor_id)
    job.to_server(controller)

    # Add MLflow Receiver for metrics streaming
    if script == "src/train_metrics.py":
        receiver = MLflowReceiver(
            tracking_uri="file:///tmp/nvflare/jobs/workdir/server/simulate_job/mlruns",
            kw_args={
                "experiment_name": "nvflare-fedavg-np-experiment",
                "run_name": "nvflare-fedavg-np-with-mlflow",
                "experiment_tags": {"mlflow.note.content": "## **NVFlare FedAvg Numpy experiment with MLflow**"},
                "run_tags": {"mlflow.note.content": "## Federated Experiment tracking with MLflow.\n"},
            },
            artifact_location="artifacts",
            events=["fed.analytix_log_stats"],
        )
        job.to_server(receiver)

    executor = ScriptRunner(
        script=script,
        launch_external_process=launch_process,
        framework=FrameworkType.NUMPY,
    )
    job.to_clients(executor)

    if export_config:
        job.export_job("/tmp/nvflare/jobs/job_config")
    else:
        job.simulator_run("/tmp/nvflare/jobs/workdir", n_clients=n_clients, gpu="0")


if __name__ == "__main__":
    main()
