# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.api import FedJob, validate_object_for_job


class BaseFedJob(FedJob):
    """Unified BaseFedJob that supports PyTorch, TensorFlow, and Scikit-learn frameworks.

    This class consolidates the previously separate PT and TF BaseFedJob implementations
    into a single unified interface that can handle all frameworks.

    Note: This class is framework-agnostic and does not contain framework-specific logic.
    Framework-specific model setup should be handled by child classes (e.g., PT/TF wrappers)
    or by the calling code (e.g., recipes).
    """

    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator: Optional[ValidationJsonGenerator] = None,
        model_selector: Optional[FLComponent] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
    ):
        """Unified BaseFedJob for PyTorch, TensorFlow, and Scikit-learn.

        Configures ValidationJsonGenerator, model selector, AnalyticsReceiver, ConvertToFedEvent.
        Framework-specific model setup should be handled by child classes or calling code (recipes).

        User must add controllers and executors.

        Args:
            name: Name of the job. Defaults to "fed_job".
            min_clients: The minimum number of clients for the job. Defaults to 1.
            mandatory_clients: Mandatory clients to run the job. Default None.
            key_metric: Metric used to determine if the model is globally best.
                If metrics are a dict, key_metric can select the metric used for global model selection.
                Defaults to "accuracy". Only used if model_selector is not provided.
            validation_json_generator: A component for generating validation results.
                If not provided, a ValidationJsonGenerator will be configured.
            model_selector: A component for selecting the best model during training.
                This event-driven component evaluates and tracks model performance across training rounds,
                determining which model is globally best based on validation metrics.
                It handles workflow events such as BEFORE_AGGREGATION and BEFORE_CONTRIBUTION_ACCEPT.
                Common implementations:
                  - IntimeModelSelector: selects based on a specified key metric
                  - SimpleIntimeModelSelector: simplified version for basic selection
                If not provided and key_metric is specified, an IntimeModelSelector will be configured
                automatically.
            convert_to_fed_event: A component to convert certain events to fed events.
                If not provided, a ConvertToFedEvent object will be created.
            analytics_receiver: Receive analytics. If not provided, framework-specific
                child classes may provide defaults (e.g., TBAnalyticsReceiver for PT/TF).
        """
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
        )

        self.comp_ids = {}

        # Validation JSON generator
        if validation_json_generator:
            validate_object_for_job("validation_json_generator", validation_json_generator, ValidationJsonGenerator)
        else:
            validation_json_generator = ValidationJsonGenerator()
        self.to_server(id="json_generator", obj=validation_json_generator)

        # Model selector
        if model_selector:
            validate_object_for_job("model_selector", model_selector, FLComponent)
            self.to_server(id="model_selector", obj=model_selector)
        elif key_metric:
            # Default to IntimeModelSelector if key_metric is provided
            from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector

            self.to_server(id="model_selector", obj=IntimeModelSelector(key_metric=key_metric))

        # Convert to fed event
        if convert_to_fed_event:
            validate_object_for_job("convert_to_fed_event", convert_to_fed_event, ConvertToFedEvent)
        else:
            convert_to_fed_event = ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE])
        self.convert_to_fed_event = convert_to_fed_event

        # Analytics receiver (child classes should provide defaults if needed)
        if analytics_receiver:
            validate_object_for_job("analytics_receiver", analytics_receiver, AnalyticsReceiver)
            self.to_server(
                id="receiver",
                obj=analytics_receiver,
            )

        # Framework-specific model setup should be handled by child classes
        # (they receive parameters directly, no need to store them here)

    def set_up_client(self, target: str):
        """Setup client components."""
        self.to(id="event_to_fed", obj=self.convert_to_fed_event, target=target)
