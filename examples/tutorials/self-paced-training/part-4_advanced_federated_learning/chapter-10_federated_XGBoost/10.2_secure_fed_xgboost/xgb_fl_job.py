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

from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.app_opt.tracking.tb.tb_writer import TBWriter
from nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader import CSVDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor
from nvflare.job_config.api import FedJob


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=3, help="Total number of training rounds")
    parser.add_argument("--nthread", type=int, default=16, help="nthread for xgboost")
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist for best perf"
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="dataset split mode, horizontal or vertical",
    )
    parser.add_argument(
        "--secure",
        type=bool,
        default=False,
        help="Whether to use secure training",
    )
    return parser.parse_args()


def _get_job_name(args) -> str:
    if args.secure:
        return f"{args.data_split_mode}_secure"
    else:
        return f"{args.data_split_mode}"


def main():
    args = define_parser()
    job_name = _get_job_name(args)
    dataset_path = args.data_root
    site_num = args.site_num
    job = FedJob(name=job_name, min_clients=site_num)
    if args.data_split_mode == "horizontal":
        data_split_mode = 0
    else:
        data_split_mode = 1
    # Define the controller workflow and send to server
    controller = XGBFedController(
        num_rounds=args.round_num,
        data_split_mode=data_split_mode,
        secure_training=args.secure,
        xgb_options={"early_stopping_rounds": 3, "use_gpus": False},
        xgb_params={
            "max_depth": 3,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "nthread": 1,
        },
        client_ranks={"site-1": 0, "site-2": 1, "site-3": 2},
        in_process=True,
    )
    job.to_server(controller, id="xgb_controller")

    # Add tensorboard receiver to server
    tb_receiver = TBAnalyticsReceiver(
        tb_folder="tb_events",
    )
    job.to_server(tb_receiver, id="tb_receiver")

    # Add executor and other components to clients
    for site_id in range(1, site_num + 1):
        # Define the executor for clients
        executor = FedXGBHistogramExecutor(
            data_loader_id="dataloader",
            metrics_writer_id="metrics_writer",
            in_process=True,
        )
        job.to(executor, f"site-{site_id}", id="executor")

        dataloader = CSVDataLoader(folder=dataset_path)
        job.to(dataloader, f"site-{site_id}", id="dataloader")

        metrics_writer = TBWriter(event_type="analytix_log_stats")
        job.to(metrics_writer, f"site-{site_id}", id="metrics_writer")

        event_to_fed = ConvertToFedEvent(
            events_to_convert=["analytix_log_stats"],
            fed_event_prefix="fed.",
        )
        job.to(event_to_fed, f"site-{site_id}", id="event_to_fed")

    # Export job config and run the job
    job.export_job("/tmp/nvflare/workspace/fedxgb_secure/train_fl/jobs/")

    # Run the job except for secure horizontal
    if args.data_split_mode == "horizontal" and args.secure:
        print("Secure horizontal is not supported in this version")
    else:
        job.simulator_run(f"/tmp/nvflare/workspace/fedxgb_secure/train_fl/works/{job_name}")


if __name__ == "__main__":
    main()
