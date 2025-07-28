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

from src.vertical_data_loader import VerticalDataLoader

from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.app_opt.tracking.tb.tb_writer import TBWriter
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController
from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor
from nvflare.job_config.api import FedJob


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_path",
        type=str,
        default="/tmp/nvflare/dataset/xgboost_higgs_vertical/{SITE_NAME}/higgs.data.csv",
        help="Path to data split files for each site",
    )
    parser.add_argument(
        "--psi_path",
        type=str,
        default="/tmp/nvflare/workspace/works/vertical_xgb_psi/{SITE_NAME}/simulate_job/{SITE_NAME}/psi/intersection.txt",
        help="Path to psi files for each site",
    )
    parser.add_argument("--site_num", type=int, default=2, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")
    return parser.parse_args()


def main():
    args = define_parser()
    data_split_path = args.data_split_path
    psi_path = args.psi_path
    site_num = args.site_num
    round_num = args.round_num
    job_name = "xgboost_vertical"
    job = FedJob(name=job_name, min_clients=site_num)

    # Define the controller workflow and send to server
    controller = XGBFedController(
        num_rounds=round_num,
        data_split_mode=1,
        secure_training=False,
        xgb_options={"early_stopping_rounds": 3, "use_gpus": False},
        xgb_params={
            "max_depth": 8,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "nthread": 16,
        },
    )
    job.to_server(controller, id="xgb_controller")

    # Add tensorboard receiver to server
    tb_receiver = TBAnalyticsReceiver(
        tb_folder="tb_events",
    )
    job.to_server(tb_receiver, id="tb_receiver")

    # Define the executor and send to clients
    executor = FedXGBHistogramExecutor(
        data_loader_id="dataloader",
        metrics_writer_id="metrics_writer",
        in_process=True,
        model_file_name="test.model.json",
    )
    job.to_clients(executor, id="xgb_hist_executor", tasks=["config", "start"])

    dataloader = VerticalDataLoader(
        data_split_path=data_split_path, psi_path=psi_path, id_col="uid", label_owner="site-1", train_proportion=0.8
    )
    job.to_clients(dataloader, id="dataloader")

    metrics_writer = TBWriter(event_type="analytix_log_stats")
    job.to_clients(metrics_writer, id="metrics_writer")

    event_to_fed = ConvertToFedEvent(
        events_to_convert=["analytix_log_stats"],
        fed_event_prefix="fed.",
    )
    job.to_clients(event_to_fed, id="event_to_fed")

    # Export job config and run the job
    job.export_job("/tmp/nvflare/workspace/jobs/")
    job.simulator_run(f"/tmp/nvflare/workspace/works/{job_name}", n_clients=site_num)


if __name__ == "__main__":
    main()
