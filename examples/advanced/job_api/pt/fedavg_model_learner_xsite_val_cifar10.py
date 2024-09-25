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

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "cifar10"))

from pt.learners.cifar10_model_learner import CIFAR10ModelLearner
from pt.networks.cifar10_nets import ModerateCNN
from pt.utils.cifar10_data_splitter import Cifar10DataSplitter
from pt.utils.cifar10_data_utils import load_cifar10_data

from nvflare.app_common.executors.model_learner_executor import ModelLearnerExecutor
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    aggregation_epochs = 4
    alpha = 0.1
    train_split_root = f"/tmp/cifar10_splits/clients{n_clients}_alpha{alpha}"  # avoid overwriting results

    job = FedAvgJob(name="cifar10_fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=ModerateCNN())

    ctrl = CrossSiteModelEval()

    load_cifar10_data()  # preload CIFAR10 data
    data_splitter = Cifar10DataSplitter(
        split_dir=train_split_root,
        num_sites=n_clients,
        alpha=alpha,
    )

    job.to(ctrl, "server")
    job.to(data_splitter, "server")

    for i in range(n_clients):
        site_name = f"site-{i + 1}"
        learner_id = job.to(
            CIFAR10ModelLearner(train_idx_root=train_split_root, aggregation_epochs=aggregation_epochs, lr=0.01),
            site_name,
            id="learner",
        )
        executor = ModelLearnerExecutor(learner_id=learner_id)
        job.to(executor, site_name)  # data splitter assumes client names start from 1

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
