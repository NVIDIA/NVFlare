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

from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.api import FedJob
from nvflare.job_config.pt.model import Wrap


class FedAvgJob(FedJob):
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

        component = ValidationJsonGenerator()
        self.to_server(id="json_generator", obj=component)

        if self.key_metric:
            component = IntimeModelSelector(key_metric=self.key_metric)
            self.to_server(id="model_selector", obj=component)

        # TODO: make different tracking receivers configurable
        component = TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
        self.to_server(id="receiver", obj=component)

        comp_ids = self.to_server(Wrap(self.initial_model))

        # Define the controller workflow and send to server
        controller = FedAvg(
            num_clients=self.n_clients,
            num_rounds=self.num_rounds,
            persistor_id=comp_ids["persistor_id"],
        )
        self.to_server(controller)

    def set_up_client(self, target: str):
        component = ConvertToFedEvent(events_to_convert=["analytix_log_stats"], fed_event_prefix="fed.")
        self.to(id="event_to_fed", obj=component, target=target)
