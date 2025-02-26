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
import os

from src.kmeans_assembler import KMeansAssembler
from utils.split_data import split_data

from nvflare import FedJob
from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.job_config.script_runner import ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/workspace/works/kmeans",
        help="work directory, default to '/tmp/nvflare/workspace/works/kmeans'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/workspace/jobs/kmeans",
        help="directory for job export, default to '/tmp/nvflare/workspace/jobs/kmeans'",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/dataset/sklearn_iris.csv",
        help="data path, default to '/tmp/nvflare/dataset/sklearn_iris.csv'",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="number of clients to simulate, default to 3",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="number of rounds, default to 5",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="how to split data among clients",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        default=1,
        help="fraction of data to use for validation, default to perform validation on all data",
    )
    return parser.parse_args()


def main():
    args = define_parser()
    # Get args
    data_path = args.data_path
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    split_mode = args.split_mode
    valid_frac = args.valid_frac
    job_name = f"sklearn_kmeans_{split_mode}_{num_clients}_clients"
    train_script = "src/kmeans_fl.py"

    # Set the output workspace and job directories
    workspace_dir = os.path.join(args.workspace_dir, job_name)
    job_dir = args.job_dir

    # Create the FedJob
    job = FedJob(name=job_name, min_clients=num_clients)

    # Define the controller workflow and send to server
    controller = ScatterAndGather(
        min_clients=num_clients,
        num_rounds=num_rounds,
        aggregator_id="aggregator",
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        train_task_name="train",
    )
    job.to_server(controller, id="scatter_and_gather")

    # Define other server components
    assembler = KMeansAssembler()
    job.to_server(assembler, id="kmeans_assembler")
    aggregator = CollectAndAssembleAggregator(assembler_id="kmeans_assembler")
    job.to_server(aggregator, id="aggregator")
    shareable_generator = FullModelShareableGenerator()
    job.to_server(shareable_generator, id="shareable_generator")
    persistor = JoblibModelParamPersistor(
        initial_params={"n_clusters": 3},
    )
    job.to_server(persistor, id="persistor")

    # Get the data split numbers and send to each client
    # generate data split
    site_indices = split_data(
        data_path,
        num_clients,
        valid_frac,
    )

    for i in range(1, num_clients + 1):
        # Define the executor and send to clients
        train_start = site_indices[i]["start"]
        train_end = site_indices[i]["end"]
        valid_start = site_indices["valid"]["start"]
        valid_end = site_indices["valid"]["end"]

        executor = ScriptRunner(
            script=train_script,
            script_args=f"--data_path {data_path} "
            f"--train_start {train_start} --train_end {train_end} "
            f"--valid_start {valid_start} --valid_end {valid_end}",
            params_exchange_format="raw",
        )
        job.to(executor, f"site-{i}", tasks=["train"])

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    job.simulator_run(workspace_dir)


if __name__ == "__main__":
    main()
