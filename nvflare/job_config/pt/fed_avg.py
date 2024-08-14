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
from typing import List, Optional

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob
from nvflare.job_config.controller_apps.deep_learning import DeepLearningControllerApp
from nvflare.job_config.defs import JobTargetType
from nvflare.job_config.pt.model import Wrap


class FedAvgJob(FedJob):
    def __init__(
        self,
        model,
        n_clients,
        num_rounds,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
    ):
        super().__init__(name, min_clients, mandatory_clients)
        server_app = DeepLearningControllerApp(key_metric=key_metric)
        self.to(server_app, JobTargetType.SERVER)

        # Define the controller workflow and send to server
        controller = FedAvg(
            num_clients=n_clients,
            num_rounds=num_rounds,
        )
        self.to(controller, JobTargetType.SERVER)

        # Define the initial global model and send to server
        self.to(Wrap(model), JobTargetType.SERVER)
