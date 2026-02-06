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

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.streaming import AnalyticsReceiver
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.base_fed_job import BaseFedJob as UnifiedBaseFedJob


class BaseFedJob(UnifiedBaseFedJob):
    """TensorFlow BaseFedJob.

    This is a backward-compatible wrapper around the unified BaseFedJob.
    For new code, consider using nvflare.job_config.base_fed_job.BaseFedJob directly with
    framework=FrameworkType.TENSORFLOW.

    Configures ValidationJsonGenerator, model selector, AnalyticsReceiver, ConvertToFedEvent.

    User must add controllers and executors.

    Args:
        model (tf.keras.Model): initial TensorFlow Model. Defaults to None.
        initial_ckpt: Absolute path to a pre-trained checkpoint file (.h5, .keras, or SavedModel dir).
            The file may not exist locally as it could be on the server.
            Note: TensorFlow can load full models from .h5/SavedModel without model.
        name (name, optional): name of the job. Defaults to "fed_job".
        min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
        mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
        key_metric (str, optional): Metric used to determine if the model is globally best.
            if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
            Defaults to "accuracy".
        validation_json_generator (ValidationJsonGenerator, optional): A component for generating validation results.
            if not provided, a ValidationJsonGenerator will be configured.
        model_selector: (FLComponent, optional): A component for selecting the best model during training.
            This event-driven component evaluates and tracks model performance across training rounds,
            handling workflow events such as BEFORE_AGGREGATION and BEFORE_CONTRIBUTION_ACCEPT.
            If not provided, an IntimeModelSelector will be configured based on key_metric.
        convert_to_fed_event: (ConvertToFedEvent, optional): A component to convert certain events to fed events.
            if not provided, a ConvertToFedEvent object will be created.
        analytics_receiver (AnalyticsReceiver | None, optional): Component for receiving analytics data.
            If not provided, no analytics tracking will be enabled. For experiment tracking (e.g., TensorBoard),
            explicitly pass a TBAnalyticsReceiver instance.
        model_persistor (optional, ModelPersistor): how to persist the model.
    """

    def __init__(
        self,
        model: tf.keras.Model = None,
        initial_ckpt: Optional[str] = None,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator: Optional[ValidationJsonGenerator] = None,
        model_selector: Optional[FLComponent] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
        analytics_receiver: Optional[AnalyticsReceiver] = None,
        model_persistor: Optional[ModelPersistor] = None,
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

        # TensorFlow-specific model setup
        # TFModel wrapper can handle: tf.keras.Model instances, dict configs, or None
        if model is not None or initial_ckpt is not None:
            self._setup_tensorflow_model(model, initial_ckpt, model_persistor)

    def _setup_tensorflow_model(
        self,
        model: Optional[tf.keras.Model],
        initial_ckpt: Optional[str],
        persistor: Optional[ModelPersistor] = None,
    ):
        """Setup TensorFlow model with persistor."""
        from nvflare.app_opt.tf.job_config.model import TFModel

        tf_model = TFModel(model=model, initial_ckpt=initial_ckpt, persistor=persistor)
        self.comp_ids["persistor_id"] = self.to_server(tf_model)
