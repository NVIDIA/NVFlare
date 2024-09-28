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

import torch.nn as nn

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob, TBAnalyticsReceiverArgs
from nvflare.app_opt.pt.job_config.model import PTFileModelPersistorArgs
from nvflare.job_config.common_components_job import (
    ConvertToFedEventArgs,
    IntimeModelSelectorArgs,
    ValidationJsonGeneratorArgs,
)


class FedAvgJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        n_clients: int,
        num_rounds: int,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator_args: Optional[ValidationJsonGeneratorArgs] = None,
        intime_model_selector_args: Optional[IntimeModelSelectorArgs] = None,
        tb_analytic_receiver_args: Optional[TBAnalyticsReceiverArgs] = None,
        convert_to_fed_event_args: Optional[ConvertToFedEventArgs] = None,
        start_round: int = 0,
        model_persistor_args: Optional[PTFileModelPersistorArgs] = None,
        model_controller_args: Optional[dict] = None,
    ):
        """PyTorch FedAvg Job.

        Configures server side FedAvg controller, persistor with initial model, and widgets.

        User must add executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model
            n_clients (int): number of clients for this job
            num_rounds (int): number of rounds for FedAvg
            name (name, optional): name of the job. Defaults to "fed_job"
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
            start_round (int, optional): The starting round number.
            model_controller_args (dict, optional): Additional arguments to be passed into underlying ModelController.
        """
        if not isinstance(initial_model, nn.Module):
            raise ValueError(f"Expected initial model to be nn.Module, but got type f{type(initial_model)}.")

        super().__init__(
            initial_model=initial_model,
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            key_metric=key_metric,
            validation_json_generator_args=validation_json_generator_args,
            intime_model_selector_args=intime_model_selector_args,
            tb_analytic_receiver_args=tb_analytic_receiver_args,
            convert_to_fed_event_args=convert_to_fed_event_args,
            model_persistor_args=model_persistor_args,
        )

        controller = FedAvg(
            num_clients=n_clients,
            num_rounds=num_rounds,
            persistor_id=self.comp_ids["persistor_id"],
            start_round=start_round,
            kwargs=model_controller_args,
        )
        self.to_server(controller)
