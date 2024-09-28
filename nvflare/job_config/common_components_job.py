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

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.job_config.api import FedJob


class ValidationJsonGeneratorArgs:
    def __init__(self, results_dir=AppConstants.CROSS_VAL_DIR, json_file_name="cross_val_results.json") -> None:
        self.results_dir = results_dir
        self.json_file_name = json_file_name


class IntimeModelSelectorArgs:
    def __init__(
        self,
        weigh_by_local_iter=False,
        aggregation_weights=None,
        validation_metric_name=MetaKey.INITIAL_METRICS,
        negate_key_metric: bool = False,
    ) -> None:
        self.weigh_by_local_iter = weigh_by_local_iter
        self.aggregation_weights = aggregation_weights
        self.validation_metric_name = validation_metric_name
        self.negate_key_metric = negate_key_metric


class ConvertToFedEventArgs:
    def __init__(self, events_to_convert=[ANALYTIC_EVENT_TYPE], fed_event_prefix="fed.") -> None:
        self.events_to_convert = events_to_convert
        self.fed_event_prefix = fed_event_prefix


class CommonComponentsJob(FedJob):
    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator_args: Optional[ValidationJsonGeneratorArgs] = None,
        intime_model_selector_args: Optional[IntimeModelSelectorArgs] = None,
        convert_to_fed_event_args: Optional[ConvertToFedEventArgs] = None,
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
        self.key_metric = key_metric

        # Initialize arguments
        self.validation_json_generator_args = validation_json_generator_args or ValidationJsonGeneratorArgs()
        self.intime_model_selector_args = intime_model_selector_args or IntimeModelSelectorArgs()
        self.convert_to_fed_event_args = convert_to_fed_event_args or ConvertToFedEventArgs()

        # Setup components
        self._setup_components()

    def set_up_client(self, target: str):
        component = ConvertToFedEvent(
            events_to_convert=self.convert_to_fed_event_args.events_to_convert,
            fed_event_prefix=self.convert_to_fed_event_args.fed_event_prefix,
        )
        self.to(id="event_to_fed", obj=component, target=target)

    def _setup_components(self):
        """Setup the components required for the job."""
        self.to_server(id="json_generator", obj=self._create_validation_json_generator())

        if self.key_metric:
            self.to_server(id="model_selector", obj=self._create_intime_model_selector())

    def _create_validation_json_generator(self):
        """Creates a ValidationJsonGenerator component."""
        return ValidationJsonGenerator(
            results_dir=self.validation_json_generator_args.results_dir,
            json_file_name=self.validation_json_generator_args.json_file_name,
        )

    def _create_intime_model_selector(self):
        """Creates an IntimeModelSelector component."""
        return IntimeModelSelector(
            key_metric=self.key_metric,
            weigh_by_local_iter=self.intime_model_selector_args.weigh_by_local_iter,
            aggregation_weights=self.intime_model_selector_args.aggregation_weights,
            validation_metric_name=self.intime_model_selector_args.validation_metric_name,
            negate_key_metric=self.intime_model_selector_args.negate_key_metric,
        )
