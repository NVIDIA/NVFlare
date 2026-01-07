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

from model import SimpleNetwork

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver

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

    # Create FedAvg recipe (without default server-side tracking)
    recipe = FedAvgRecipe(
        name="fedavg_mlflow_client",
        min_clients=args.n_clients,
        num_rounds=5,
        initial_model=SimpleNetwork(),
        train_script="client.py",
    )

    # Add site-specific MLflow tracking for each client
    for i in range(args.n_clients):
        site_name = f"site-{i + 1}"
        tracking_uri = f"file://{args.work_dir}/{site_name}/mlruns"

        receiver = MLflowReceiver(
            tracking_uri=tracking_uri,
            events=[ANALYTIC_EVENT_TYPE],  # Track local events (not federated)
            kw_args={
                "experiment_name": "nvflare-fedavg-experiment",
                "run_name": f"nvflare-fedavg-{site_name}",
                "experiment_tags": {"mlflow.note.content": f"## **NVFlare FedAvg experiment - {site_name}**"},
                "run_tags": {"mlflow.note.content": f"## Site-specific tracking for {site_name}.\n"},
            },
        )

        # Add receiver to this specific client
        recipe.job.to(receiver, site_name, id="mlflow_receiver")

    # Run or export
    if args.export_config:
        print(f"Exporting job config...{args.job_configs}/fedavg_mlflow_client")
        recipe.export(args.job_configs)
    else:
        recipe.run(workspace=args.work_dir, log_config=args.log_config)
