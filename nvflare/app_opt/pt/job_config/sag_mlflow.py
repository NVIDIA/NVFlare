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
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob, TBAnalyticsReceiverArgs
from nvflare.app_opt.pt.job_config.model import PTFileModelPersistorArgs
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.app_opt.tracking.mlflow.mlflow_writer import MLflowWriter
from nvflare.job_config.common_components_job import (
    ConvertToFedEventArgs,
    IntimeModelSelectorArgs,
    ValidationJsonGeneratorArgs,
)


class MLflowReceiverArgs:
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        kw_args: Optional[dict] = None,
        artifact_location: Optional[str] = None,
        events=None,
        buffer_flush_time=1,
    ):
        self.tracking_uri = tracking_uri
        self.kw_args = kw_args
        self.artifact_location = artifact_location
        self.events = events
        self.buffer_flush_time = buffer_flush_time


class InTimeAccumulateWeightedAggregatorArgs:
    def __init__(
        self,
        exclude_vars: Union[str, Dict[str, str], None] = None,
        aggregation_weights: Union[Dict[str, Any], Dict[str, Dict[str, Any]], None] = None,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHTS,
        weigh_by_local_iter: bool = True,
    ) -> None:
        self.exclude_vars = exclude_vars
        self.aggregation_weights = aggregation_weights
        self.expected_data_kind = expected_data_kind
        self.weigh_by_local_iter = weigh_by_local_iter


class ScatterAndGatherArgs:
    def __init__(
        self,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
    ) -> None:
        self.start_round = start_round
        self.wait_time_after_min_received = wait_time_after_min_received
        self.train_task_name = train_task_name
        self.train_timeout = train_timeout
        self.ignore_result_error = ignore_result_error
        self.allow_empty_global_weights = allow_empty_global_weights
        self.task_check_period = task_check_period
        self.persist_every_n_rounds = persist_every_n_rounds


class SAGMLFlowJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        num_rounds: int,
        min_clients: int,
        name: str = "fed_job",
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
        validation_json_generator_args: Optional[ValidationJsonGeneratorArgs] = None,
        intime_model_selector_args: Optional[IntimeModelSelectorArgs] = None,
        tb_analytic_receiver_args: Optional[TBAnalyticsReceiverArgs] = None,
        convert_to_fed_event_args: Optional[ConvertToFedEventArgs] = None,
        model_persistor_args: Optional[PTFileModelPersistorArgs] = None,
        mlflow_receiver_args: Optional[MLflowReceiverArgs] = None,
        intime_accumulate_weighted_aggregator_args: Optional[InTimeAccumulateWeightedAggregatorArgs] = None,
        scatter_and_gather_args: Optional[ScatterAndGatherArgs] = None,
    ):
        """PyTorch ScatterAndGather with MLFlow Job.

        Configures server side ScatterAndGather controller, persistor with initial model, and widgets.

        User must add executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model
            num_rounds (int): number of rounds for FedAvg
            name (name, optional): name of the job. Defaults to "fed_job"
            min_clients (int): the minimum number of clients for the job.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
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

        mlflow_receiver_args = mlflow_receiver_args if mlflow_receiver_args else MLflowReceiverArgs()
        intime_accumulate_weighted_aggregator_args = (
            intime_accumulate_weighted_aggregator_args
            if intime_accumulate_weighted_aggregator_args
            else InTimeAccumulateWeightedAggregatorArgs()
        )
        scatter_and_gather_args = scatter_and_gather_args if scatter_and_gather_args else ScatterAndGatherArgs()

        shareable_generator_id = self.to_server(FullModelShareableGenerator(), id="shareable_generator")
        aggregator_id = self.to_server(
            InTimeAccumulateWeightedAggregator(
                exclude_vars=intime_accumulate_weighted_aggregator_args.exclude_vars,
                expected_data_kind=intime_accumulate_weighted_aggregator_args.expected_data_kind,
                aggregation_weights=intime_accumulate_weighted_aggregator_args.aggregation_weights,
                weigh_by_local_iter=intime_accumulate_weighted_aggregator_args.weigh_by_local_iter,
            ),
            id="aggregator",
        )

        component = MLflowReceiver(
            tracking_uri=mlflow_receiver_args.tracking_uri,
            kw_args=mlflow_receiver_args.kw_args,
            artifact_location=mlflow_receiver_args.artifact_location,
            events=mlflow_receiver_args.events,
            buffer_flush_time=mlflow_receiver_args.buffer_flush_time,
        )
        self.to_server(id="mlflow_receiver_with_tracking_uri", obj=component)

        controller = ScatterAndGather(
            min_clients=min_clients,
            num_rounds=num_rounds,
            start_round=scatter_and_gather_args.start_round,
            wait_time_after_min_received=scatter_and_gather_args.wait_time_after_min_received,
            aggregator_id=aggregator_id,
            persistor_id=self.comp_ids["persistor_id"],
            shareable_generator_id=shareable_generator_id,
            train_task_name=scatter_and_gather_args.train_task_name,
            train_timeout=scatter_and_gather_args.train_timeout,
            ignore_result_error=scatter_and_gather_args.ignore_result_error,
            allow_empty_global_weights=scatter_and_gather_args.allow_empty_global_weights,
            task_check_period=scatter_and_gather_args.task_check_period,
            persist_every_n_rounds=scatter_and_gather_args.persist_every_n_rounds,
        )
        self.to_server(controller)

    def set_up_client(self, target: str):
        super().set_up_client(target)

        ml_flow_writer = MLflowWriter(event_type="event_type")
        self.to(id="log_writer", obj=ml_flow_writer, target=target)
