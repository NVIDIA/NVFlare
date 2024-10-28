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

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.app_opt.tracking.mlflow.mlflow_writer import MLflowWriter


class SAGMLFlowJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        n_clients: int,
        num_rounds: int,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        tracking_uri=None,
        kwargs=None,
        artifact_location=None,
    ):
        """PyTorch ScatterAndGather with MLFlow Job.

        Configures server side ScatterAndGather controller, persistor with initial model, and widgets.

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
            kwargs: kwargs dict
        """
        super().__init__(initial_model, name, min_clients, mandatory_clients, key_metric)

        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = self.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = self.to_server(
            InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator"
        )

        component = MLflowReceiver(tracking_uri=tracking_uri, kw_args=kwargs, artifact_location=artifact_location)
        self.to_server(id="mlflow_receiver_with_tracking_uri", obj=component)

        controller = ScatterAndGather(
            min_clients=n_clients,
            num_rounds=num_rounds,
            wait_time_after_min_received=10,
            aggregator_id=aggregator_id,
            persistor_id=self.comp_ids["persistor_id"],
            shareable_generator_id=shareable_generator_id,
        )
        self.to_server(controller)

    def set_up_client(self, target: str):
        super().set_up_client(target)

        ml_flow_writer = MLflowWriter(event_type="event_type")
        self.to(id="log_writer", obj=ml_flow_writer, target=target)
