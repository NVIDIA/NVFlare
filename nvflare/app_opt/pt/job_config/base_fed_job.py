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

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.base_fed_job import BaseFedJob as UnifiedBaseFedJob


class BaseFedJob(UnifiedBaseFedJob):
    """PyTorch BaseFedJob.

    This is a backward-compatible wrapper around the unified BaseFedJob.

    Configures ValidationJsonGenerator, model selector, AnalyticsReceiver, ConvertToFedEvent.

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
        model_selector: (FLComponent | None, optional): A component for selecting the best model during training.
            This event-driven component evaluates and tracks model performance across training rounds,
            handling workflow events such as BEFORE_AGGREGATION and BEFORE_CONTRIBUTION_ACCEPT.
            If not provided, an IntimeModelSelector will be configured based on key_metric.
        convert_to_fed_event: (ConvertToFedEvent | None, optional): A component to convert certain events to fed events.
            if not provided, a ConvertToFedEvent object will be created.
        analytics_receiver (AnalyticsReceiver | None, optional): Component for receiving analytics data.
            If not provided, no analytics tracking will be enabled. For experiment tracking (e.g., TensorBoard),
            explicitly pass a TBAnalyticsReceiver instance.
        model_persistor (ModelPersistor | None, optional): how to persist the model.
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
        model_selector: Optional[FLComponent] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        model_persistor: Optional[ModelPersistor] = None,
        model_locator: Optional[ModelLocator] = None,
    ):
        # Call the unified BaseFedJob
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            key_metric=key_metric,
            validation_json_generator=validation_json_generator,
            model_selector=model_selector,
            convert_to_fed_event=convert_to_fed_event,
            analytics_receiver=analytics_receiver,
        )

        # PyTorch-specific model setup
        if initial_model is not None:
            if not isinstance(initial_model, nn.Module):
                raise TypeError(
                    f"initial_model must be an instance of nn.Module, but got {type(initial_model).__name__}"
                )
            self._setup_pytorch_model(initial_model, model_persistor, model_locator)

    def _setup_pytorch_model(
        self,
        model: nn.Module,
        persistor: Optional[ModelPersistor] = None,
        locator: Optional[ModelLocator] = None,
    ):
        """Setup PyTorch model with persistor and locator."""
        from nvflare.app_opt.pt.job_config.model import PTModel

        pt_model = PTModel(model=model, persistor=persistor, locator=locator)
        self.comp_ids.update(self.to_server(pt_model))
