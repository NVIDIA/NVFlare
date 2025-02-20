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
from enum import Enum
from typing import List

import numpy as np
from src.kmeans_assembler import KMeansAssembler
from src.kmeans_learner import KMeansLearner

from nvflare import FedJob
from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.app_opt.sklearn.sklearn_executor import SKLearnExecutor


class SplitMethod(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    SQUARE = "square"
    EXPONENTIAL = "exponential"


def get_split_ratios(site_num: int, split_method: SplitMethod):
    if split_method == SplitMethod.UNIFORM:
        ratio_vec = np.ones(site_num)
    elif split_method == SplitMethod.LINEAR:
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif split_method == SplitMethod.SQUARE:
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif split_method == SplitMethod.EXPONENTIAL:
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError(f"Split method {split_method.name} not implemented!")

    return ratio_vec


def split_num_proportion(n, site_num, split_method: SplitMethod) -> List[int]:
    split = []
    ratio_vec = get_split_ratios(site_num, split_method)
    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def assign_data_index_to_sites(
    data_size: int,
    valid_fraction: float,
    num_sites: int,
    split_method: SplitMethod = SplitMethod.UNIFORM,
) -> dict:
    if valid_fraction > 1.0:
        raise ValueError("validation percent should be less than or equal to 100% of the total data")
    elif valid_fraction < 1.0:
        valid_size = int(round(data_size * valid_fraction, 0))
        train_size = data_size - valid_size
    else:
        valid_size = data_size
        train_size = data_size

    site_sizes = split_num_proportion(train_size, num_sites, split_method)
    split_data_indices = {
        "valid": {"start": 0, "end": valid_size},
    }
    for site in range(num_sites):
        site_id = site + 1
        if valid_fraction < 1.0:
            idx_start = valid_size + sum(site_sizes[:site])
            idx_end = valid_size + sum(site_sizes[: site + 1])
        else:
            idx_start = sum(site_sizes[:site])
            idx_end = sum(site_sizes[: site + 1])
        split_data_indices[site_id] = {"start": idx_start, "end": idx_end}

    return split_data_indices


def get_file_line_count(input_path: str) -> int:
    count = 0
    with open(input_path, "r") as fp:
        for i, _ in enumerate(fp):
            count += 1
    return count


def split_data(
    data_path: str,
    num_clients: int,
    valid_frac: float,
    split_method: SplitMethod = SplitMethod.UNIFORM,
):
    size_total_file = get_file_line_count(data_path)
    site_indices = assign_data_index_to_sites(size_total_file, valid_frac, num_clients, split_method)
    return site_indices


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
        SplitMethod(split_mode),
    )

    for i in range(1, num_clients + 1):
        # Define the executor and send to clients
        runner = SKLearnExecutor(learner_id="kmeans_learner")
        job.to(runner, f"site-{i}", tasks=["train"])

        learner = KMeansLearner(
            data_path=data_path,
            train_start=site_indices[i]["start"],
            train_end=site_indices[i]["end"],
            valid_start=site_indices["valid"]["start"],
            valid_end=site_indices["valid"]["end"],
            random_state=0,
        )
        job.to(learner, f"site-{i}", id="kmeans_learner")

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    job.simulator_run(workspace_dir)


if __name__ == "__main__":
    main()
