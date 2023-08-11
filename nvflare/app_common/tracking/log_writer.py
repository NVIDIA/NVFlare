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

from abc import ABC, abstractmethod
from typing import Any
from nvflare.apis.analytix import AnalyticsDataType

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metric_exchange.metric_exchanger import MetricExchanger
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender


class LogWriter(FLComponent, ABC):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE, metrics_exchanger_id: str = None):
        """Base class for log writer with sender.

        If the metrics_exchanger_id is provided, 

        Args:
            event_type (str, optional): Used for AnalyticsSender when metrics_exchanger_id is not provided. Defaults to ANALYTIC_EVENT_TYPE.
            metrics_exchanger_id (str, optional): If this is provided, expects MetricsExchanger with this id. Defaults to None.
        """
        super().__init__()
        self.metrics_exchanger_id = metrics_exchanger_id
        if self.metrics_exchanger_id:
            self.sender = None
        else:
            self.sender = self.load_log_sender(event_type)

    def load_log_sender(self, event_type: str = ANALYTIC_EVENT_TYPE) -> AnalyticsSender:
        return AnalyticsSender(event_type, self.get_writer_name())

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            if self.metrics_exchanger_id is None:
                # provide engine to sender if using AnalyticsSender
                self.sender.engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            if self.metrics_exchanger_id:
                # get MetricsExchanger as sender if using MetricsExchanger
                engine = fl_ctx.get_engine()
                self.sender = engine.get_component(self.metrics_exchanger_id)
                if self.sender is None:
                    self.task_panic("Cannot load MetricExchanger!", fl_ctx=fl_ctx)

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        if isinstance(self.sender, MetricExchanger):
            self.sender.log(key=key, value=value, data_type=data_type, **kwargs)
        elif isinstance(self.sender, AnalyticsSender):
            self.sender.add(tag=key, value=value, data_type=data_type, **kwargs)

    @abstractmethod
    def get_writer_name(self) -> LogWriterName:
        pass
