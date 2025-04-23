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

from src.network import SimpleNetwork

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.job_config.script_runner import ScriptRunner

WORKSPACE = "/tmp/nvflare/jobs/workdir"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_clients", type=int, default=2)
    parser.add_argument("-j", "--job_configs", type=str, nargs="?", default="/tmp/nvflare/jobs")
    parser.add_argument("-w", "--work_dir", type=str, nargs="?", default=WORKSPACE)
    parser.add_argument("-e", "--export_config", action="store_true", help="config only mode, export config")
    parser.add_argument("-l", "--log_config", type=str, default="concise")

    return parser.parse_args()


if __name__ == "__main__":

    args = define_parser()
    n_clients = args.n_clients
    job_configs = args.job_configs
    work_dir = args.work_dir
    export_config = args.export_config
    log_config = args.log_config

    num_rounds = 5

    train_script = "src/train_script.py"

    job_name = "fedavg"

    # create a FedAvg job but not registered the default components for fed_event converter and analytics receiver.
    # As we are going to add ourselves.

    job = FedAvgJob(
        name=job_name,
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        convert_to_fed_event=False,
        analytics_receiver=False,
    )

    # Add a MLFlow Receiver component to the Client site

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, f"site-{i + 1}")

        tracking_uri = f"file://{work_dir}/site-{i + 1}/mlruns"
        receiver = MLflowReceiver(
            tracking_uri=tracking_uri,
            events=[ANALYTIC_EVENT_TYPE],
            kw_args={
                "experiment_name": "nvflare-fedavg-experiment",
                "run_name": "nvflare-fedavg-with-mlflow",
                "experiment_tags": {"mlflow.note.content": "## **NVFlare FedAvg experiment with MLflow**"},
                "run_tags": {"mlflow.note.content": "## Federated Experiment tracking with MLflow.\n"},
            },
        )

        job.to(receiver, target="site-{i + 1}")

    if export_config:
        print(f"Exporting job config...{job_configs}/{job_name}")
        job.export_job(job_root=job_configs)
    else:
        job.simulator_run(workspace=work_dir, log_config=log_config)
