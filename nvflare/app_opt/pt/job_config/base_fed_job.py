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

from torch import nn as nn

from nvflare import FedJob
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver


class BaseFedJob(FedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
    ):
        """PyTorch BaseFedJob.

        Configures server side FedAvg controller, persistor with initial model, and widgets.

        User must add executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model
            name (name, optional): name of the job. Defaults to "fed_job"
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
        super().__init__(name, min_clients, mandatory_clients)
        self.key_metric = key_metric
        self.initial_model = initial_model

        component = ValidationJsonGenerator()
        self.to_server(id="json_generator", obj=component)

        if self.key_metric:
            component = IntimeModelSelector(key_metric=self.key_metric)
            self.to_server(id="model_selector", obj=component)

        # TODO: make different tracking receivers configurable
        component = TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
        self.to_server(id="receiver", obj=component)

        self.comp_ids = self.to_server(PTModel(initial_model))

    def set_up_client(self, target: str):
        component = ConvertToFedEvent(events_to_convert=["analytix_log_stats"], fed_event_prefix="fed.")
        self.to(id="event_to_fed", obj=component, target=target)
