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
from typing import Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender


class LogWriter(FLComponent, ABC):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE, metrics_sender_id: str = None):
        super().__init__()
        self.event_type = event_type
        self.metrics_sender_id = metrics_sender_id
        self.sender = None
        self.engine = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            if self.metrics_sender_id:
                self.sender = engine.get_component(self.metrics_sender_id)
                if self.sender is None:
                    self.system_panic("Cannot load MetricsSender!", fl_ctx=fl_ctx)
                self.sender.writer = self.get_writer_name()
            else:
                self.sender = AnalyticsSender(self.event_type, self.get_writer_name())
                self.sender.engine = engine

    def write(self, tag: str, value, data_type: AnalyticsDataType, global_step: Optional[int] = None, **kwargs):
        """Writes a record.

        Args:
            tag (str): Tag name
            value: Value to send
            data_type (AnalyticsDataType): Data type of the value being sent
            global_step (optional, int): Global step value.

        Raises:
            TypeError: global_step must be an int
        """
        self.sender.add(tag=tag, value=value, data_type=data_type, global_step=global_step, **kwargs)

    @abstractmethod
    def get_writer_name(self) -> LogWriterName:
        pass

    def get_default_metric_data_type(self) -> AnalyticsDataType:
        return AnalyticsDataType.METRICS
