# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import logging
from typing import Optional

from client import SyntheticDataExecutor

from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.feature_election.controller import FeatureElectionController
from nvflare.job_config.api import FedJob

logger = logging.getLogger(__name__)


def create_feature_election_job(
    job_name: str = "feature_election_synthetic",
    num_clients: int = 3,
    freedom_degree: float = 0.5,
    aggregation_mode: str = "weighted",
    num_rounds: int = 5,
    auto_tune: bool = False,
    tuning_rounds: int = 4,
    fs_method: str = "lasso",
    eval_metric: str = "f1",
    split_strategy: str = "stratified",
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 20,
    n_redundant: int = 30,
    n_repeated: int = 30,
    export_dir: Optional[str] = None,
) -> FedJob:
    job = FedJob(name=job_name)

    controller = FeatureElectionController(
        freedom_degree=freedom_degree,
        aggregation_mode=aggregation_mode,
        min_clients=num_clients,
        num_rounds=num_rounds,
        task_name="feature_election",
        auto_tune=auto_tune,
        tuning_rounds=tuning_rounds,
    )
    job.to_server(controller)
    job.to_server(ValidationJsonGenerator())

    executor = SyntheticDataExecutor(
        fs_method=fs_method,
        eval_metric=eval_metric,
        num_clients=num_clients,
        split_strategy=split_strategy,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        task_name="feature_election",
    )

    # FIXED: Uses to_clients instead of to_client
    job.to_clients(executor)

    if export_dir:
        job.export_job(export_dir)

    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", default="feature_election_synthetic")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--freedom-degree", type=float, default=0.5)
    parser.add_argument("--aggregation-mode", default="weighted")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--tuning-rounds", type=int, default=4)
    parser.add_argument("--fs-method", default="lasso")
    parser.add_argument("--eval-metric", default="f1")
    parser.add_argument("--split-strategy", default="stratified")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--n-informative", type=int, default=20)
    parser.add_argument("--n-redundant", type=int, default=30)
    parser.add_argument("--n-repeated", type=int, default=30)
    parser.add_argument("--workspace", default="/tmp/nvflare/feature_election")
    parser.add_argument("--threads", type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    job = create_feature_election_job(
        job_name=args.job_name,
        num_clients=args.num_clients,
        freedom_degree=args.freedom_degree,
        aggregation_mode=args.aggregation_mode,
        num_rounds=args.num_rounds,
        auto_tune=args.auto_tune,
        tuning_rounds=args.tuning_rounds,
        fs_method=args.fs_method,
        eval_metric=args.eval_metric,
        split_strategy=args.split_strategy,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_repeated=args.n_repeated,
        export_dir=args.export_dir,
    )

    job.simulator_run(
        workspace=args.workspace,
        n_clients=args.num_clients,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
