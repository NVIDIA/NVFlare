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

from nvflare.app_opt.pt.job_config.model import PTFileModelPersistorArgs, PTModel
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.common_components_job import (
    CommonComponentsJob,
    ConvertToFedEventArgs,
    IntimeModelSelectorArgs,
    ValidationJsonGeneratorArgs,
)


class TBAnalyticsReceiverArgs:
    def __init__(self, tb_folder="tb_events", events: Optional[List[str]] = ["fed.analytix_log_stats"]):
        self.tb_folder = tb_folder
        self.events = events


class BaseFedJob(CommonComponentsJob):
    def __init__(
        self,
        initial_model: nn.Module = None,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator_args: Optional[ValidationJsonGeneratorArgs] = None,
        intime_model_selector_args: Optional[IntimeModelSelectorArgs] = None,
        tb_analytic_receiver_args: Optional[TBAnalyticsReceiverArgs] = None,
        convert_to_fed_event_args: Optional[ConvertToFedEventArgs] = None,
        model_persistor_args: Optional[PTFileModelPersistorArgs] = None,
    ):
        """PyTorch BaseFedJob.

        Configures ValidationJsonGenerator, IntimeModelSelector, TBAnalyticsReceiver, ConvertToFedEvent.

        User must add controllers and executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model. Defaults to None.
            name (name, optional): name of the job. Defaults to "fed_job".
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
        super().__init__(
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            validation_json_generator_args=validation_json_generator_args,
            intime_model_selector_args=intime_model_selector_args,
            convert_to_fed_event_args=convert_to_fed_event_args,
        )
        self.key_metric = key_metric
        self.initial_model = initial_model
        self.comp_ids = {}

        # Initialize arguments
        self.tb_analytic_receiver_args = tb_analytic_receiver_args or TBAnalyticsReceiverArgs()

        self.to_server(
            id="receiver",
            obj=TBAnalyticsReceiver(
                tb_folder=self.tb_analytic_receiver_args.tb_folder,
                events=self.tb_analytic_receiver_args.events,
            ),
        )

        if initial_model:
            self.comp_ids.update(
                self.to_server(PTModel(model=initial_model, model_persistor_args=model_persistor_args))
            )
