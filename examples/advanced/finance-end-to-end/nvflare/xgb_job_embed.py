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

from xgb_embed_data_loader import CreditCardEmbedDataLoader

from nvflare import FedJob
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor


def main():
    args = define_parser()

    site_names = args.sites
    work_dir = args.work_dir
    job_name = args.job_name
    root_dir = args.input_dir
    file_postfix = args.file_postfix

    num_rounds = 10
    early_stopping_rounds = 10
    xgb_params = {
        "max_depth": 8,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "nthread": 16,
    }

    job = FedJob(name=job_name)

    # Define the controller workflow and send to server
    controller = XGBFedController(
        num_rounds=num_rounds,
        data_split_mode=0,
        secure_training=False,
        xgb_params=xgb_params,
        xgb_options={"early_stopping_rounds": early_stopping_rounds},
    )
    job.to(controller, "server")

    # Add clients
    for site_name in site_names:
        executor = FedXGBHistogramExecutor(data_loader_id="data_loader")
        job.to(executor, site_name)
        data_loader = CreditCardEmbedDataLoader(root_dir=root_dir, file_postfix=file_postfix)
        job.to(data_loader, site_name, id="data_loader")

    if work_dir:
        print("work_dir=", work_dir)
        job.export_job(work_dir)

    if not args.config_only:
        job.simulator_run(work_dir)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--sites",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated site names",
    )
    parser.add_argument(
        "-n",
        "--job_name",
        type=str,
        nargs="?",
        default="xgb_job",
        help="job name, default to xgb_job",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        nargs="?",
        default="/tmp/nvflare/jobs/xgb/workdir",
        help="work directory, default to '/tmp/nvflare/jobs/xgb/workdir'",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="?",
        default="",
        help="root directory for input data",
    )
    parser.add_argument(
        "-p",
        "--file_postfix",
        type=str,
        nargs="?",
        default="_combined.csv",
        help="file ending postfix, such as '.csv', or '_combined.csv'",
    )

    parser.add_argument("-co", "--config_only", action="store_true", help="config only mode, will not run simulator")

    return parser.parse_args()


if __name__ == "__main__":
    main()
