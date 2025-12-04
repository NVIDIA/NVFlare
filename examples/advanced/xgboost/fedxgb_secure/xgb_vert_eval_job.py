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

from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
from nvflare.app_opt.xgboost.histogram_based_v2.fed_eval_executor import FedXGBEvalExecutor
from nvflare.job_config.api import FedJob


def define_parser():
    parser = argparse.ArgumentParser(description="Federated XGBoost Secure Vertical Evaluation Job")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument(
        "--train_workspace_path",
        type=str,
        default="/tmp/nvflare/workspace/fedxgb_secure/train_fl/works/vertical_secure",
        help="Path to the trained model workspace root",
    )
    return parser.parse_args()


def main():
    args = define_parser()
    dataset_path = args.data_root
    site_num = args.site_num
    train_workspace_path = args.train_workspace_path

    # Create job
    job_name = "xgb_vert_eval"
    job = FedJob(name=job_name, min_clients=site_num)

    # Define the evaluation controller
    # secure_training flag has impact over the XGB inner logic
    # set to False explicitly to avoid confusion (although it is evaluation)
    controller = XGBFedController(
        num_rounds=1,
        data_split_mode=1,
        secure_training=False,
        xgb_options={},
        xgb_params={"nthread": 1},
        client_ranks={f"site-{i + 1}": i for i in range(site_num)},
        in_process=True,
    )
    job.to_server(controller, id="xgb_controller")

    # Add executor and other components to clients
    for site_id in range(1, site_num + 1):
        # Define the evaluation executor for clients
        executor = FedXGBEvalExecutor(
            data_loader_id="dataloader",
            train_workspace_path=train_workspace_path,
        )
        job.to(executor, f"site-{site_id}", id="executor")

        dataloader = CSVDataLoader(folder=dataset_path)
        job.to(dataloader, f"site-{site_id}", id="dataloader")

    # Export job config
    job.export_job("/tmp/nvflare/workspace/fedxgb_secure/eval_fl/jobs/")

    # Run the job
    job.simulator_run(workspace="/tmp/nvflare/workspace/fedxgb_secure/eval_fl/works/secure_vert_eval")


if __name__ == "__main__":
    main()
