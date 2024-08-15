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
from nvflare.job_config.api import ExecutorApp, JobTargetType
from nvflare.job_config.controller_apps.deep_learning import DeepLearningControllerApp
from nvflare.job_config.executor_apps.basic import BasicExecutorApp
from nvflare.job_config.generic_job import GenericJob
from nvflare.job_config.pt.model import PTModel


class FedAvgJob(GenericJob):
    def __init__(
        self,
        initial_model,
        n_clients,
        num_rounds,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
    ):
        super().__init__(name, min_clients, mandatory_clients)
        self.key_metric = key_metric
        self.initial_model = initial_model
        self.num_rounds = num_rounds
        self.n_clients = n_clients

    def set_up_server(self):
        server_app = DeepLearningControllerApp(key_metric=self.key_metric)
        self.to_server(server_app)

        comp_ids = self.to_server(PTModel(self.initial_model))

        # Define the controller workflow and send to server
        controller = FedAvg(
            num_clients=self.n_clients,
            num_rounds=self.num_rounds,
            persistor_id=comp_ids["persistor_id"],
        )
        self.to_server(controller)

    def get_executor_app(self, client: str) -> ExecutorApp:
        return BasicExecutorApp()
