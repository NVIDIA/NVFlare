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
import time
import traceback
from typing import Callable, Dict, Optional, Tuple

from net import Net

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.utils.math_utils import parse_compare_criteria, parse_compare_operator
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    RESP_MAX_WAIT_TIME,
    START_ROUND,
)
from nvflare.app_common.workflows.wf_comm.wf_spec import WF
from nvflare.security.logging import secure_format_traceback

update_model = FLModelUtils.update_model


# FedAvg Workflow


class FedAvg(WF):
    def __init__(
        self,
        min_clients: int,
        num_rounds: int,
        output_path: str,
        start_round: int = 1,
        stop_cond: str = None,
        model_selection_rule: str = None,
        resp_max_wait_time: float = 5.0,
    ):
        super(FedAvg, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.output_path = output_path
        self.min_clients = min_clients
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.current_round = start_round
        self.resp_max_wait_time = resp_max_wait_time
        self.best_model: Optional[FLModel] = None
        if stop_cond:
            self.stop_criteria = parse_compare_criteria(stop_cond)
        else:
            self.stop_criteria = None

        if model_selection_rule:
            self.metric_comp_rule = parse_compare_operator(model_selection_rule)
        else:
            self.metric_comp_rule = None

    def run(self):
        self.logger.info("start Fed Avg Workflow\n \n")

        start = self.start_round
        end = self.start_round + self.num_rounds

        model = self.init_model()
        for current_round in range(start, end):

            self.logger.info(f"Round {current_round}/{self.num_rounds} started. {start=}, {end=}")
            self.current_round = current_round

            if self.should_stop(model.metrics, self.stop_criteria):
                self.logger.info(f"stop at {current_round}/{self.num_rounds}, early stop condition satisfied.")
                break

            self.scatter(model, current_round)

            self.logger.info("gather and aggregate")
            aggr_result = self.gather_and_aggr(self.in_time_aggr_fn)

            self.logger.info(f"aggregate metrics = {aggr_result.metrics}")

            model = update_model(model, aggr_result)

            self.select_best_model(model)

        self.save_model(self.best_model, self.output_path)

        self.logger.info("end Fed Avg Workflow\n \n")

    def init_model(self):
        net = Net()
        model = FLModel(params=net.state_dict(), params_type=ParamsType.FULL)
        return model

    def scatter(self, model: FLModel, current_round):
        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            RESP_MAX_WAIT_TIME: self.resp_max_wait_time,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: self.start_round,
            DATA: model,
        }

        # (2) broadcast and wait
        self.flare_comm.broadcast(msg_payload)

    def gather_and_aggr(self, in_time_aggr_fn: Callable):
        responses = 0
        start = None
        aggr_model = None
        aggr_params_helper = WeightedAggregationHelper()
        aggr_metrics_helper = WeightedAggregationHelper()
        helpers = (aggr_params_helper, aggr_metrics_helper)

        while True:
            if responses >= self.min_clients:
                return aggr_model
            else:
                time.sleep(0.2)

            max_timeout = self.resp_max_wait_time if start else None
            self.logger.warning(f"{max_timeout=}, {responses=}, {self.min_clients=}")
            try:
                item = self.flare_comm.wait_one(max_timeout)
            except RuntimeError as e:
                self.logger.error(traceback.format_exc())
                break

            task_name, site_name, model = item
            aggr_model = in_time_aggr_fn(helpers, aggr_model, site_name, model)
            start = time.time() if start is None else start
            responses += 1

        if responses < self.min_clients:
            raise RuntimeError(
                f"not enough responses {responses} compare with min responses requirement {self.min_clients} within the max allowed time {self.resp_max_wait_time} seconds"
            )
        else:
            return aggr_model

    def in_time_aggr_fn(self, helpers: Tuple, prev_mode: FLModel, site_name: str, fl_model: FLModel) -> FLModel:

        self.logger.info("Fed Avg in time aggregate \n")
        if not fl_model:
            raise RuntimeError("model must not be None")

        self.logger.info(f"aggregating update at round {self.current_round}")

        self.logger.info(f"site={site_name}  {fl_model.metrics=}")

        if prev_mode is None:
            self.logger.info(f"aggr_metrics={fl_model.metrics}")
            return fl_model

        try:
            params_type = fl_model.params_type
            aggr_params_helper, aggr_metrics_helper = helpers

            aggr_params_helper.add(
                data=fl_model.params,
                weight=self.current_round,
                contributor_name=site_name,
                contribution_round=self.current_round,
            )

            aggr_metrics_helper.add(
                data=fl_model.metrics,
                weight=self.current_round,
                contributor_name=site_name,
                contribution_round=self.current_round,
            )

            aggr_params = aggr_params_helper.get_result()
            aggr_metrics = aggr_metrics_helper.get_result()

            self.logger.info(f"{aggr_metrics=}")

            aggr_result = FLModel(
                params=aggr_params,
                params_type=params_type,
                metrics=aggr_metrics,
                meta={
                    "num_rounds_aggregated": 1 + (self.current_round - self.start_round),
                    "current_round": self.current_round,
                },
            )
            return aggr_result

        except Exception as e:
            raise RuntimeError(f"Exception in aggregate call: {secure_format_traceback()}")

    def select_best_model(self, curr_model: FLModel):
        if self.best_model is None:
            self.best_model = curr_model
            return

        if self.metric_comp_rule is None:
            return
        metric, op_fn = self.metric_comp_rule

        self.logger.info("compare models")
        if self.is_curr_mode_better(self.best_model, curr_model, metric, op_fn):
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
