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
from typing import Dict, List, Optional

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Feature, Histogram, HistogramType, StatisticConfig, Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.app_common.executors.common_executor import CommonExecutor
from nvflare.app_common.executors.statistics.statistics_client_executor import StatisticsClientExecutor
from nvflare.app_common.statistics.numeric_stats import filter_numeric_features
from nvflare.app_common.statistics.statisitcs_objects_decomposer import fobs_registration
from nvflare.app_common.statistics.statistics_config_utils import get_feature_bin_range
from nvflare.fuel.utils import fobs

"""
    StatisticsExecutor is client-side executor that perform local statistics generation and communication to
    FL Server global statistics controller.
    The actual local statistics calculation would delegate to Statistics spec implementor.
"""


class StatisticsExecutor(CommonExecutor):
    def __init__(
        self,
        generator_id: str,
        precision=4,
    ):
        """

        Args:
            generator_id:  Id of the statistics component

            precision: number of precision digits

        """

        super().__init__()
        self.generator_id = generator_id
        self.precision = precision

    def get_client_executor(self, fl_ctx: FLContext) -> ClientExecutor:
        client_executor = StatisticsClientExecutor(self.generator_id, self.precision)
        client_executor.initialize(fl_ctx)
        return client_executor

