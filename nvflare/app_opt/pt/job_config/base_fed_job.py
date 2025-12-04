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
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.base_fed_job import BaseFedJob as UnifiedBaseFedJob
from nvflare.job_config.script_runner import FrameworkType


class BaseFedJob(UnifiedBaseFedJob):
    """PyTorch BaseFedJob.

    This is a backward-compatible wrapper around the unified BaseFedJob.
    For new code, consider using nvflare.job_config.base_fed_job.BaseFedJob directly with
    framework=FrameworkType.PYTORCH.

    Configures ValidationJsonGenerator, IntimeModelSelector, AnalyticsReceiver, ConvertToFedEvent.

    User must add controllers and executors.

    Args:
        initial_model (nn.Module, optional): initial PyTorch Model. Defaults to None.
        name (str, optional): name of the job. Defaults to "fed_job".
        min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
        mandatory_clients (list[str] | None, optional): mandatory clients to run the job. Default None.
        key_metric (str, optional): Metric used to determine if the model is globally best.
            if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
            Defaults to "accuracy".
        validation_json_generator (ValidationJsonGenerator | None, optional): A component for generating validation results.
            if not provided, a ValidationJsonGenerator will be configured.
        intime_model_selector: (IntimeModelSelector | None, optional): A component for select the model.
            if not provided, an IntimeModelSelector will be configured.
        convert_to_fed_event: (ConvertToFedEvent | None, optional): A component to covert certain events to fed events.
            if not provided, a ConvertToFedEvent object will be created.
        analytics_receiver (bool | AnalyticsReceiver | None, optional): Receive analytics.
            If not provided, a TBAnalyticsReceiver will be configured.
        model_persistor (ModelPersistor | None, optional): how to persistor the model.
        model_locator (ModelLocator | None, optional): how to locate the model.
    """

    def __init__(
        self,
        initial_model: nn.Module = None,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator: Optional[ValidationJsonGenerator] = None,
        intime_model_selector: Optional[IntimeModelSelector] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        model_persistor: Optional[ModelPersistor] = None,
        model_locator: Optional[ModelLocator] = None,
    ):
        # Call the unified BaseFedJob with PyTorch-specific settings
        super().__init__(
            initial_model=initial_model,
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            key_metric=key_metric,
            validation_json_generator=validation_json_generator,
            intime_model_selector=intime_model_selector,
            convert_to_fed_event=convert_to_fed_event,
            analytics_receiver=analytics_receiver,
            model_persistor=model_persistor,
            model_locator=model_locator,
            framework=FrameworkType.PYTORCH,
        )
