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
from nvflare.app_common.tracking.tracker_types import LogWriterName


class LogWriterForMetricsExchanger(FLComponent, ABC):
    def __init__(self, metrics_exchanger_id: str):
        """Base class for log writer for MetricsExchanger.

        Args:
            metrics_exchanger_id (str, optional): Expects MetricsExchanger with this id. Defaults to None.
        """
        super().__init__()
        self.metrics_exchanger_id = metrics_exchanger_id
        self.sender = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.sender = engine.get_component(self.metrics_exchanger_id)
            if self.sender is None:
                self.task_panic("Cannot load MetricsExchanger!", fl_ctx=fl_ctx)

    def send_log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        self.sender.log(key=key, value=value, data_type=data_type, **kwargs)

    @abstractmethod
    def get_writer_name(self) -> LogWriterName:
        pass
