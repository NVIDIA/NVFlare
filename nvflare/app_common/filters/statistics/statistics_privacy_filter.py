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

from typing import List, Optional, Union

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import MetricConfig
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.filters.statistics.metrics_privacy_filter import MetricsPrivacyFilter
from nvflare.fuel.utils import fobs


class StatisticsPrivacyFilter(DXOFilter):
    def __init__(self, result_filter_ids: List[str]):
        super().__init__(supported_data_kinds=[DataKind.STATISTICS], data_kinds_to_filter=[DataKind.STATISTICS])
        self.result_filter_ids = result_filter_ids

    def get_filters(self, result_filter_ids: List[str], fl_ctx: FLContext) -> List[MetricsPrivacyFilter]:
        filters = []
        for filter_id in result_filter_ids:
            c = fl_ctx.get_engine().get_component(filter_id)
            if not isinstance(c, MetricsPrivacyFilter):
                msg = "component identified by {} type {} is not type of MetricsPrivacyFilter".format(
                    filter_id, type(c)
                )
                raise ValueError(msg)
            filters.append(c)
        return filters

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        if dxo.data_kind == DataKind.STATISTICS:
            self.log_info(fl_ctx, "start StatisticsPrivacyFilter")
            filters: List[MetricsPrivacyFilter] = self.get_filters(self.result_filter_ids, fl_ctx)

            client_name = fl_ctx.get_identity_name()
            self.log_info(fl_ctx, f"apply StatisticPrivacyFilter for client {client_name}")
            dxo1 = self.filter_stats_metrics(dxo, client_name, filters)
            self.log_info(fl_ctx, "end StatisticsPrivacyFilter")
            return dxo1

    def filter_stats_metrics(self, dxo: DXO, client_name: str, filters: List[MetricsPrivacyFilter]) -> Optional[DXO]:
        client_result = dxo.data
        metric_task = client_result[StC.METRIC_TASK_KEY]
        metrics = fobs.loads(client_result[metric_task])
        metric_modified = False
        for f in filters:
            (metrics, modified) = f.apply(metrics, client_name)
            metric_modified = metric_modified or modified

        dxo1 = dxo
        if metric_modified:
            client_result[metric_task] = fobs.dumps(metrics)
            dxo1 = DXO(data_kind=DataKind.STATISTICS, data=client_result)

        return dxo1
