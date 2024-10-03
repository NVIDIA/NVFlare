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

from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.common_components_job import CommonComponentsJob


class BaseFedJob(CommonComponentsJob):
    def __init__(
        self,
        initial_model: nn.Module = None,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        model_persistor: Optional[ModelPersistor] = None,
        model_locator: Optional[ModelLocator] = None,
    ):
        """PyTorch BaseFedJob.

        Configures ValidationJsonGenerator, IntimeModelSelector, AnalyticsReceiver, ConvertToFedEvent.

        User must add controllers and executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model. Defaults to None.
            name (name, optional): name of the job. Defaults to "fed_job".
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
            analytics_receiver (AnlyticsReceiver, optional): Receive analytics.
                If not provided, a TBAnalyticsReceiver will be configured.
            model_persistor (optional, ModelPersistor): how to persistor the model.
            model_locator (optional, ModelLocator): how to locate the model.
        """
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            validation_json_generator=ValidationJsonGenerator(),
            intime_model_selector=IntimeModelSelector(key_metric=key_metric) if key_metric else None,
            convert_to_fed_event=ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE]),
        )

        self.initial_model = initial_model
        self.comp_ids = {}

        analytics_receiver = analytics_receiver if analytics_receiver else TBAnalyticsReceiver()

        self.to_server(
            id="receiver",
            obj=analytics_receiver,
        )

        if initial_model:
            self.comp_ids.update(
                self.to_server(PTModel(model=initial_model, persistor=model_persistor, locator=model_locator))
            )
