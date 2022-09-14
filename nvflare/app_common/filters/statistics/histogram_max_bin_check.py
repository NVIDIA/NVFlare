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
from typing import Optional, Tuple, Union

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.fuel.utils import fobs


class HistogramMaxBinCheck(DXOFilter):

    def process_dxo(self, dxo: DXO, inputs: Shareable, fl_ctx: FLContext) -> Optional[DXO]:

        if StC.STATS_MIN in inputs and StC.STATS_MAX in inputs:
            global_min_value = inputs[StC.STATS_MIN][dataset_name][feature_name]
            global_max_value = inputs[StC.STATS_MAX][dataset_name][feature_name]
            hist_config: dict = metric_config.config
            num_of_bins: int = self.get_number_of_bins(feature_name, hist_config)
            bin_range: List[int] = self.get_bin_range(feature_name, global_min_value, global_max_value, hist_config)
            item_count = self.stats_generator.count(dataset_name, feature_name)
            if num_of_bins >= item_count * self.max_bins_percent:
                raise ValueError(
                    f"number of bins: {num_of_bins} needs to be smaller than item count: {round(item_count * self.max_bins_percent)} "
                    f"for feature '{feature_name}' in dataset '{dataset_name}'"
                )