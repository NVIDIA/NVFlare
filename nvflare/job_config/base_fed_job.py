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

from typing import Any, Dict, List, Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.api import FedJob, validate_object_for_job
from nvflare.job_config.script_runner import FrameworkType


class BaseFedJob(FedJob):
    """Unified BaseFedJob that supports PyTorch, TensorFlow, and Scikit-learn frameworks.

    This class consolidates the previously separate PT and TF BaseFedJob implementations
    into a single unified interface that can handle all frameworks.
    """

    def __init__(
        self,
        initial_model: Any = None,
        initial_params: Optional[Dict] = None,
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
        framework: FrameworkType = FrameworkType.PYTORCH,
    ):
        """Unified BaseFedJob for PyTorch, TensorFlow, and Scikit-learn.

        Configures ValidationJsonGenerator, IntimeModelSelector, AnalyticsReceiver, ConvertToFedEvent.

        User must add controllers and executors.

        Args:
            initial_model: Initial model object. Can be:
                - nn.Module for PyTorch
                - tf.keras.Model for TensorFlow
                - None if using initial_params for sklearn
            initial_params: Initial model parameters (dict). Used for sklearn.
                If provided, initial_model should be None.
            name: Name of the job. Defaults to "fed_job".
            min_clients: The minimum number of clients for the job. Defaults to 1.
            mandatory_clients: Mandatory clients to run the job. Default None.
            key_metric: Metric used to determine if the model is globally best.
                If metrics are a dict, key_metric can select the metric used for global model selection.
                Defaults to "accuracy".
            validation_json_generator: A component for generating validation results.
                If not provided, a ValidationJsonGenerator will be configured.
            intime_model_selector: A component for selecting the model.
                If not provided, an IntimeModelSelector will be configured.
            convert_to_fed_event: A component to convert certain events to fed events.
                If not provided, a ConvertToFedEvent object will be created.
            analytics_receiver: Receive analytics.
                If not provided, a TBAnalyticsReceiver will be configured.
            model_persistor: How to persist the model. Framework-specific defaults will be used if not provided.
            model_locator: How to locate the model. Only used for PyTorch.
            framework: The framework type (PYTORCH, TENSORFLOW, or RAW for sklearn).
                Defaults to PYTORCH.
        """
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
        )

        self.initial_model = initial_model
        self.initial_params = initial_params
        self.framework = framework
        self.comp_ids = {}

        # Validate that only one of initial_model or initial_params is provided
        if initial_model is not None and initial_params is not None:
            raise ValueError("Cannot provide both initial_model and initial_params. Choose one based on framework.")

        # Validation JSON generator
        if validation_json_generator:
            validate_object_for_job("validation_json_generator", validation_json_generator, ValidationJsonGenerator)
        else:
            validation_json_generator = ValidationJsonGenerator()
        self.to_server(id="json_generator", obj=validation_json_generator)

        # Intime model selector
        if intime_model_selector:
            validate_object_for_job("intime_model_selector", intime_model_selector, IntimeModelSelector)
            self.to_server(id="model_selector", obj=intime_model_selector)
        elif key_metric:
            self.to_server(id="model_selector", obj=IntimeModelSelector(key_metric=key_metric))

        # Convert to fed event
        if convert_to_fed_event:
            validate_object_for_job("convert_to_fed_event", convert_to_fed_event, ConvertToFedEvent)
        else:
            convert_to_fed_event = ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE])
        self.convert_to_fed_event = convert_to_fed_event

        # Analytics receiver
        if analytics_receiver:
            validate_object_for_job("analytics_receiver", analytics_receiver, AnalyticsReceiver)
        else:
            analytics_receiver = TBAnalyticsReceiver()

        self.to_server(
            id="receiver",
            obj=analytics_receiver,
        )

        # Handle initial model/params based on framework
        if framework == FrameworkType.PYTORCH:
            if initial_model is not None:
                self._setup_pytorch_model(initial_model, model_persistor, model_locator)
        elif framework == FrameworkType.TENSORFLOW:
            if initial_model is not None:
                self._setup_tensorflow_model(initial_model, model_persistor)
        elif framework == FrameworkType.RAW:
            # Sklearn case - handled by the recipe using JoblibModelParamPersistor
            pass
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def _setup_pytorch_model(
        self,
        model: Any,
        persistor: Optional[ModelPersistor] = None,
        locator: Optional[ModelLocator] = None,
    ):
        """Setup PyTorch model with persistor and locator."""
        from nvflare.app_opt.pt.job_config.model import PTModel

        pt_model = PTModel(model=model, persistor=persistor, locator=locator)
        self.comp_ids.update(self.to_server(pt_model))

    def _setup_tensorflow_model(self, model: Any, persistor: Optional[ModelPersistor] = None):
        """Setup TensorFlow model with persistor."""
        from nvflare.app_opt.tf.job_config.model import TFModel

        tf_model = TFModel(model=model, persistor=persistor)
        self.comp_ids["persistor_id"] = self.to_server(tf_model)

    def set_up_client(self, target: str):
        """Setup client components."""
        self.to(id="event_to_fed", obj=self.convert_to_fed_event, target=target)
