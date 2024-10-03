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

from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.api import FedJob, validate_object_for_job


class CommonComponentsJob(FedJob):
    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        validation_json_generator: Optional[ValidationJsonGenerator] = None,
        intime_model_selector: Optional[IntimeModelSelector] = None,
        convert_to_fed_event: Optional[ConvertToFedEvent] = None,
    ):
        """Common components job.

        Configures ValidationJsonGenerator, IntimeModelSelector, ConvertToFedEvent.

        Args:
            name (name, optional): name of the job. Defaults to "fed_job".
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
        super().__init__(name, min_clients, mandatory_clients)

        # Initialize arguments
        if validation_json_generator:
            validate_object_for_job("validation_json_generator", validation_json_generator, ValidationJsonGenerator)
            self.to_server(id="json_generator", obj=validation_json_generator)
        if intime_model_selector:
            validate_object_for_job("intime_model_selector", intime_model_selector, IntimeModelSelector)
            self.to_server(id="model_selector", obj=intime_model_selector)
        if convert_to_fed_event:
            validate_object_for_job("convert_to_fed_event", convert_to_fed_event, ConvertToFedEvent)

        self.convert_to_fed_event = convert_to_fed_event

    def set_up_client(self, target: str):
        if self.convert_to_fed_event:
            self.to(id="event_to_fed", obj=self.convert_to_fed_event, target=target)
