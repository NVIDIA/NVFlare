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

import tensorflow as tf

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.api import FedJob, validate_object_for_job


class BaseFedJob(FedJob):
    def __init__(
        self,
        initial_model: tf.keras.Model = None,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator: Optional[ValidationJsonGenerator] = None,
        intime_model_selector: Optional[IntimeModelSelector] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        model_persistor: Optional[ModelPersistor] = None,
    ):
        """TensorFlow BaseFedJob.

        Configures ValidationJsonGenerator, IntimeModelSelector, TBAnalyticsReceiver, ConvertToFedEvent.

        User must add controllers and executors.

        Args:
            initial_model (tf.keras.Model): initial TensorFlow Model. Defaults to None.
            name (name, optional): name of the job. Defaults to "fed_job".
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
            validation_json_generator (ValidationJsonGenerator, optional): A component for generating validation results.
                if not provided, a ValidationJsonGenerator will be configured.
            intime_model_selector: (IntimeModelSelector, optional): A component for select the model.
                if not provided, an IntimeModelSelector will be configured.
            convert_to_fed_event: (ConvertToFedEvent, optional): A component to covert certain events to fed events.
                if not provided, a ConvertToFedEvent object will be created.
            analytics_receiver (AnlyticsReceiver, optional): Receive analytics.
                If not provided, a TBAnalyticsReceiver will be configured.
            model_persistor (optional, ModelPersistor): how to persistor the model.
        """
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
        )

        self.initial_model = initial_model
        self.comp_ids = {}

        if validation_json_generator:
            validate_object_for_job("validation_json_generator", validation_json_generator, ValidationJsonGenerator)
        else:
            validation_json_generator = ValidationJsonGenerator()
        self.to_server(id="json_generator", obj=validation_json_generator)

        if intime_model_selector:
            validate_object_for_job("intime_model_selector", intime_model_selector, IntimeModelSelector)
            self.to_server(id="model_selector", obj=intime_model_selector)
        elif key_metric:
            self.to_server(id="model_selector", obj=IntimeModelSelector(key_metric=key_metric))

        if convert_to_fed_event:
            validate_object_for_job("convert_to_fed_event", convert_to_fed_event, ConvertToFedEvent)
        else:
            convert_to_fed_event = ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE])
        self.convert_to_fed_event = convert_to_fed_event

        if analytics_receiver:
            validate_object_for_job("analytics_receiver", analytics_receiver, AnalyticsReceiver)
        else:
            analytics_receiver = TBAnalyticsReceiver()

        self.to_server(
            id="receiver",
            obj=analytics_receiver,
        )

        if initial_model:
            self.comp_ids["persistor_id"] = self.to_server(TFModel(model=initial_model, persistor=model_persistor))

    def set_up_client(self, target: str):
        self.to(id="event_to_fed", obj=self.convert_to_fed_event, target=target)
