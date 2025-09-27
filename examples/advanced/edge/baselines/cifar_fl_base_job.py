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

import torch
import torch.nn as nn
from torch.nn import functional as F

from nvflare import FedJob
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner


class Cifar10ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    n_clients = 16
    num_rounds = 10
    train_script = "cifar_fl_base.py"

    job = FedJob(name="cifar10_fl_base")

    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    persistor_id = job.to_server(PTFileModelPersistor(model=Cifar10ConvNet()), id="persistor")
    aggregator_id = job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind="WEIGHTS"), id="aggregator")
    controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",
    )
    job.to_server(controller)
    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",
        )
        job.to(executor, f"site-{i + 1}")

    job.export_job("/tmp/nvflare/jobs/cifar10")
    job.simulator_run("/tmp/nvflare/workspaces/cifar10_fl/", gpu="0")
