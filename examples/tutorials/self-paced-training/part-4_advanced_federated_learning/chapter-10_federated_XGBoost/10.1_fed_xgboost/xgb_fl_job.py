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
from nvflare.job_config.api import FedJob

ALGO_DIR_MAP = {
    "bagging": "tree-based",
    "cyclic": "tree-based",
    "histogram": "histogram-based",
}


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/dataset/xgb_dataset",
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=30, help="Total number of training rounds")
    parser.add_argument(
        "--training_algo", type=str, default="histogram", choices=list(ALGO_DIR_MAP.keys()), help="Training algorithm"
    )
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
    return parser.parse_args()


def _get_job_name(args) -> str:
    return f"fedxgb_{args.site_num}_sites_{args.data_split_mode}_{args.training_algo}"


def _get_data_path(args) -> str:
    return f"{args.data_root}/{args.data_split_mode}_xgb_data"


def main():
    args = define_parser()
    job_name = _get_job_name(args)
    dataset_path = _get_data_path(args)
    site_num = args.site_num
    job = FedJob(name=job_name, min_clients=site_num)
    if args.data_split_mode == "horizontal":
        data_split_mode = 0
    else:
        data_split_mode = 1
    # Define the controller workflow and send to server
    if args.training_algo == "histogram":
        from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController

        controller = XGBFedController(
            num_rounds=args.round_num,
            data_split_mode=data_split_mode,
            secure_training=False,
            xgb_options={"early_stopping_rounds": 3, "use_gpus": False},
            xgb_params={
                "max_depth": 3,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "nthread": 1,
            },
        )

        from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor

        executor = FedXGBHistogramExecutor(
            data_loader_id="dataloader",
            metrics_writer_id="metrics_writer",
        )
        # Add tensorboard receiver to server
        tb_receiver = TBAnalyticsReceiver(
            tb_folder="tb_events",
        )
        job.to_server(tb_receiver, id="tb_receiver")
    elif args.training_algo == "bagging":
        from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

        controller = ScatterAndGather(
            min_clients=args.site_num,
            num_rounds=args.round_num,
            start_round=0,
            aggregator_id="aggregator",
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
            wait_time_after_min_received=0,
            train_timeout=0,
            allow_empty_global_weights=True,
            task_check_period=0.01,
            persist_every_n_rounds=0,
            snapshot_every_n_rounds=0,
        )
        from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor

        persistor = XGBModelPersistor(save_name="xgboost_model.json")
        from nvflare.app_opt.xgboost.tree_based.shareable_generator import XGBModelShareableGenerator

        shareable_generator = XGBModelShareableGenerator()
        from nvflare.app_opt.xgboost.tree_based.bagging_aggregator import XGBBaggingAggregator

        aggregator = XGBBaggingAggregator()
        job.to_server(persistor, id="persistor")
        job.to_server(shareable_generator, id="shareable_generator")
        job.to_server(aggregator, id="aggregator")
    elif args.training_algo == "cyclic":
        from nvflare.app_common.workflows.cyclic_ctl import CyclicController

        controller = CyclicController(
            num_rounds=int(args.round_num / args.site_num),
            task_assignment_timeout=60,
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
            task_check_period=0.01,
            persist_every_n_rounds=0,
            snapshot_every_n_rounds=0,
        )
        from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor

        persistor = XGBModelPersistor(save_name="xgboost_model.json", load_as_dict=False)
        from nvflare.app_opt.xgboost.tree_based.shareable_generator import XGBModelShareableGenerator

        shareable_generator = XGBModelShareableGenerator()
        job.to_server(persistor, id="persistor")
        job.to_server(shareable_generator, id="shareable_generator")
    # send controller to server
    job.to_server(controller, id="xgb_controller")

    # Add executor and other components to clients
    for site_id in range(1, site_num + 1):
        if args.training_algo in ["bagging", "cyclic"]:
            num_client_bagging = 1
            if args.training_algo == "bagging":
                num_client_bagging = args.site_num

            from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor

            executor = FedXGBTreeExecutor(
                data_loader_id="dataloader",
                training_mode=args.training_algo,
                num_client_bagging=num_client_bagging,
                num_local_parallel_tree=1,
                local_subsample=1,
                local_model_path="model.json",
                global_model_path="model_global.json",
                learning_rate=0.1,
                objective="binary:logistic",
                max_depth=3,
                lr_scale=1,
                eval_metric="auc",
                tree_method="hist",
                nthread=1,
            )
        job.to(executor, f"site-{site_id}")

        dataloader = CSVDataLoader(folder=dataset_path)
        job.to(dataloader, f"site-{site_id}", id="dataloader")

        if args.training_algo in ["histogram"]:
            metrics_writer = TBWriter(event_type="analytix_log_stats")
            job.to(metrics_writer, f"site-{site_id}", id="metrics_writer")

            event_to_fed = ConvertToFedEvent(
                events_to_convert=["analytix_log_stats"],
                fed_event_prefix="fed.",
            )
            job.to(event_to_fed, f"site-{site_id}", id="event_to_fed")

    # Export job config and run the job
    job.export_job("/tmp/nvflare/workspace/fedxgb/jobs/")
    job.simulator_run(f"/tmp/nvflare/workspace/fedxgb/works/{job_name}")


if __name__ == "__main__":
    main()
