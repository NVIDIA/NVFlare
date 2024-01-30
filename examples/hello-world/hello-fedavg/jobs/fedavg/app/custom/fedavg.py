# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import sys
from typing import Callable, Dict, Optional

from net import Net
from nvflare.apis.wf_controller import WFController
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.security.logging import secure_format_traceback

update_model = FLModelUtils.update_model


# FedAvg Workflow


class FedAvg(WFController):
    def __init__(
            self,
            min_clients: int,
            num_rounds: int,
            output_path: str,
            start_round: int = 1,
            stop_cond: str = None,
            resp_max_wait_time: float = 5,
    ):
        super(FedAvg, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.task_name = "train"
        self.output_path = output_path
        self.min_clients = min_clients
        self.resp_max_wait_time = resp_max_wait_time
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.current_round = start_round
        self.best_model: Optional[FLModel] = None
        self.aggr_params_helper = WeightedAggregationHelper()
        self.aggr_metrics_helper = WeightedAggregationHelper()
        self.params_type: Optional[ParamsType] = None
        if stop_cond:
            self.stop_criteria = parse_compare_criteria(stop_cond)
        else:
            self.stop_criteria = None

    def run(self):
        self.logger.info("start Fed Avg Workflow\n \n")
        start = self.start_round
        end = self.start_round + self.num_rounds

        model = self.init_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for current_round in range(start, end):

            self.logger.info(f"Round {current_round}/{self.num_rounds} started. {start=}, {end=}")
            self.current_round = current_round

            if self.should_stop(model.metrics, self.stop_criteria):
                self.logger.info(f"stop at {current_round}/{self.num_rounds}, early stop condition satisfied.")
                break

            # no callback
            sag_results = self.prepare_broadcast_and_wait(self.task_name, model)
            aggr_result = self.aggr_fn(sag_results)

            # # with callback
            # self.broadcast_and_wait(task_name=self.task_name, model=model, callback=self.callback)
            # aggr_result = self.aggr_fn()

            self.logger.info(f"aggregate metrics = {aggr_result.metrics}")
            model = update_model(model, aggr_result)
            self.select_best_model(model)

        self.save_model(self.best_model, self.output_path)

        self.logger.info("end Fed Avg Workflow\n \n")

    def init_model(self):
        net = Net()
        model = FLModel(params=net.state_dict(), params_type=ParamsType.FULL)
        return model

    def prepare_broadcast_and_wait(self, task_name, model: FLModel, callback=None):
        # (2) broadcast and wait
        model.current_round = self.current_round
        results = self.broadcast_and_wait(
            task_name=task_name, min_responses=self.min_clients, data=model, callback=callback
        )
        if callback is None:
            return results
        else:
            return None

    def callback(self, data, topic):
        self.intime_agg_fn(data, self.aggr_params_helper, self.aggr_metrics_helper)

    def intime_agg_fn(self, data, aggr_params_helper, aggr_metrics_helper):
        self.logger.info("\n fed avg intime_aggregate \n")

        if not data:
            raise RuntimeError("input is None or empty")
        task_name, task_result = next(iter(data.items()))

        try:
            for site, fl_model in task_result.items():
                if self.params_type is None:
                    self.params_type = fl_model.params_type

                aggr_params_helper.add(
                    data=fl_model.params,
                    weight=self.current_round,
                    contributor_name=site,
                    contribution_round=self.current_round,
                )

                self.logger.info(f"site={site}  {fl_model.metrics=}")

                aggr_metrics_helper.add(
                    data=fl_model.metrics,
                    weight=self.current_round,
                    contributor_name=site,
                    contribution_round=self.current_round,
                )

        except Exception as e:
            raise RuntimeError(f"Exception in aggregate call: {secure_format_traceback()}")

    def intime_aggr_fn(self, sag_result: Optional[Dict[str, Dict[str, FLModel]]] = None) -> FLModel:
        if self.callback:
            return self.get_aggr_result(self.aggr_params_helper, self.aggr_metrics_helper)
        else:
            raise ValueError("callback function needs to be defined")

    def aggr_fn(self, sag_result: Optional[Dict[str, Dict[str, FLModel]]] = None) -> FLModel:
        self.logger.info("fed avg aggregate \n")

        if not sag_result:
            raise RuntimeError("input is None or empty")

        # we only have one task
        task_name, task_result = next(iter(sag_result.items()))
        self.logger.info(f"aggregating {len(task_result)} update(s) at round {self.current_round}")

        try:
            aggr_params_helper = WeightedAggregationHelper()
            aggr_metrics_helper = WeightedAggregationHelper()
            params_type = None
            for site, fl_model in task_result.items():
                if params_type is None:
                    params_type = fl_model.params_type

                aggr_params_helper.add(
                    data=fl_model.params,
                    weight=self.current_round,
                    contributor_name=site,
                    contribution_round=self.current_round,
                )

                self.logger.info(f"site={site}  {fl_model.metrics=}")

                aggr_metrics_helper.add(
                    data=fl_model.metrics,
                    weight=self.current_round,
                    contributor_name=site,
                    contribution_round=self.current_round,
                )

            return self.get_aggr_result(aggr_params_helper, aggr_metrics_helper)

        except Exception as e:
            raise RuntimeError(f"Exception in aggregate call: {secure_format_traceback()}")

    def select_best_model(self, curr_model: FLModel):
        if self.best_model is None:
            self.best_model = curr_model
            return

        if self.stop_criteria:
            metric, _, op_fn = self.stop_criteria
            self.logger.info("compare models")
            if self.is_curr_mode_better(self.best_model, curr_model, metric, op_fn):
                self.best_model = curr_model
        else:
            self.best_model = curr_model

    def save_model(self, model: FLModel, file_path: str):
        pass

    def should_stop(self, metrics: Optional[Dict] = None, stop_criteria: Optional[str] = None):
        self.logger.info(f"stop_criteria, metrics = {stop_criteria=}, {metrics=}")
        if stop_criteria is None or metrics is None:
            return False

        key, target, op_fn = stop_criteria
        value = metrics.get(key, None)

        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        return op_fn(value, target)

    def is_curr_mode_better(
            self, best_model: FLModel, curr_model: FLModel, target_metric: str, op_fn: Callable
    ) -> bool:
        curr_metrics = curr_model.metrics
        if curr_metrics is None:
            return False
        if target_metric not in curr_metrics:
            return False

        best_metrics = best_model.metrics
        return op_fn(curr_metrics.get(target_metric), best_metrics.get(target_metric))

    def get_aggr_result(self, aggr_params_helper, aggr_metrics_helper):
        aggr_params = aggr_params_helper.get_result()
        aggr_metrics = aggr_metrics_helper.get_result()

        aggr_result = FLModel(
            params=aggr_params,
            params_type=self.params_type,
            metrics=aggr_metrics,
            meta={
                "num_rounds_aggregated": 1 + (self.current_round - self.start_round),
                "current_round": self.current_round,
            },
        )
        return aggr_result
