# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import random
from typing import Optional, List, Dict

from nvflare.apis.dxo import DataKind, DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.numeric_stats import filter_numeric_features
from nvflare.app_common.statistics.stats_def import Feature
from nvflare.app_common.validation_exception import ValidationException


class StatisticsExecutor(Executor):
    def __init__(
            self,
            generator_id: str,
            min_count: int,
            min_random: float,
            max_random: float,
    ):
        super().__init__()
        self.generator_id = generator_id
        self.min_count = min_count
        self.stats_generator: Optional[Statistics] = None
        self.max_random = max_random
        self.min_random = min_random

    def metric_functions(self) -> dict:
        return \
            {StC.STATS_COUNT: self.stats_generator.get_count,
             StC.STATS_SUM: self.stats_generator.get_sum,
             StC.STATS_MEAN: self.stats_generator.get_mean,
             StC.STATS_STDDEV: self.stats_generator.get_stddev,
             StC.STATS_VAR: self.stats_generator.get_variance_with_mean,
             StC.STATS_HISTOGRAM: self.stats_generator.get_histogram,
             StC.STATS_MAX: self._get_max_value,
             StC.STATS_MIN: self._get_min_value,
             }

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
        # do nothing
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            self.stats_generator = engine.get_component(self.generator_id)
            if not isinstance(self.stats_generator, Statistics):
                raise TypeError(
                    f"statistics generator must implement Statistics type. Got: {type(self.stats_generator)}")
            self.stats_generator.initialize(engine.get_all_components(), fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner initialize exception: {e}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        try:
            result = self.client_exec(task_name, shareable, fl_ctx, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if result:
                dxo = DXO(data_kind=DataKind.ANALYTIC, data=result)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        result = Shareable()
        metrics_result = {}
        if task_name == StC.FED_STATS_TASK:
            ds_features = self.get_numeric_features(fl_ctx)
            self._validate(client_name, ds_features, shareable, fl_ctx)
            metric_task = shareable.get(StC.METRIC_TASK_KEY)
            metrics = shareable.get(StC.STATS_TARGET_METRICS)
            for metric in metrics:
                fn = self.metric_functions()[metric]
                metrics_result[metric] = {}
                for ds_name in ds_features:
                    metrics_result[metric][ds_name] = {}
                    features: List[Feature] = ds_features[ds_name]
                    for feature in features:
                        metrics_result[metric][ds_name][feature.feature_name] = \
                            fn(ds_name, feature.feature_name, shareable, fl_ctx)

            result.set_header(StC.METRIC_TASK_KEY, metric_task)
            if metric_task == StC.STATS_1st_METRICS:
                result.set_header(StC.STATS_FEATURES, ds_features)

            result[metric_task] = metrics_result

        return result

    def get_numeric_features(self, fl_ctx) -> Dict[str, List[Feature]]:
        ds_features: Dict[str, List[Feature]] = self.stats_generator.get_features(fl_ctx)
        return filter_numeric_features(ds_features)

    def _validate(self, client_name: str, ds_features: Dict[str, List[Feature]], shareable: Shareable, fl_ctx: FLContext):
        for ds_name in ds_features:
            features = ds_features[ds_name]
            for feature in features:
                feature_name = feature.feature_name
                count = self.stats_generator.get_count(ds_name, feature.feature_name, shareable, fl_ctx)
                if count < self.min_count:
                    raise ValidationException(
                        f" dataset {ds_name} feature{feature_name} item count is "
                        f"less than required minimum count {self.min_count} for client {client_name} ")


    def _get_max_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """
           get randomized max value
        """
        client_max_value = self.stats_generator.get_max_value(dataset_name, feature_name, inputs, fl_ctx)

        if client_max_value == 0  :
           max_value = 1
        elif 0 < client_max_value < 1e-3:
            max_value = 1
        elif 0 > client_max_value and abs(client_max_value) < 1e-3:
            max_value = 0
        else:
            r = random.uniform(self.max_random, self.max_random)
            if client_max_value > 0:
                max_value = client_max_value * (1 + r)
            else:
                max_value = client_max_value * (1 - r)

        return max_value

    def _get_min_value(self,
                      dataset_name: str,
                      feature_name: str,
                      inputs: Shareable,
                      fl_ctx: FLContext) -> float:
        """
           get randomized min value
        """
        client_min_value = self.stats_generator.get_min_value(dataset_name, feature_name, inputs, fl_ctx)

        if client_min_value == 0  :
            min_value = -1
        elif 0 < client_min_value < 1e-3:
            min_value = 0
        elif 0 > client_min_value and abs(client_min_value) < 1e-3:
            min_value = -1
        else:
            r = random.uniform(self.max_random, self.max_random)
            if client_min_value > 0:
                min_value = client_min_value * (1 - r)
            else:
                min_value = client_min_value * (1 + r)

        return min_value

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.stats_generator:
                self.stats_generator.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"Statistics generator finalize exception: {e}")
