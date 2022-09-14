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
from typing import Optional, Tuple

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.fuel.utils import fobs


class AddNoiseToMinMax(DXOFilter):

    def __init__(self,
                 min_noise_level: float,
                 max_noise_level: float):
        super().__init__()
        self.noise_level = (min_noise_level, max_noise_level)
        self.noise_generators = {
            StC.STATS_MIN: AddNoiseToMinMax._get_min_value,
            StC.STATS_MAX: AddNoiseToMinMax._get_max_value
        }

    @staticmethod
    def _get_min_value(local_min_value: float, noise_level: Tuple):
        r = random.uniform(noise_level[0], noise_level[1])
        if local_min_value == 0:
            min_value = -(1 - r) * 1e-5
        else:
            if local_min_value > 0:
                min_value = local_min_value * (1 - r)
            else:
                min_value = local_min_value * (1 + r)

        return min_value

    @staticmethod
    def _get_max_value(local_max_value: float, noise_level: Tuple):
        r = random.uniform(noise_level[0], noise_level[1])
        if local_max_value == 0:
            max_value = (1 + r) * 1e-5
        else:
            if local_max_value > 0:
                max_value = local_max_value * (1 + r)
            else:
                max_value = local_max_value * (1 - r)

        return max_value

    def generate_noise(self, metrics: dict, metric) -> dict:
        noise_gen = self.noise_generators[metric]
        for ds_name in metrics[metric]:
            for feature_name in metrics[metric][ds_name]:
                local_value = metrics[metric][ds_name][feature_name]
                noise_value = noise_gen(local_value, self.noise_level)
                metrics[metric][ds_name][feature_name] = noise_value
        return metrics

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Optional[DXO]:
        if dxo.data_kind == DataKind.STATISTICS:
            client_result = dxo.data
            metric_task = client_result[StC.METRIC_TASK_KEY]
            metrics = fobs.loads(client_result[metric_task])
            for metric in metrics:
                if metric in self.noise_generators:
                    metrics = self.generate_noise(metrics, metric)
            client_result[metric_task] = fobs.dumps(metrics)
            return DXO(data_kind=DataKind.STATISTICS, data=client_result)
