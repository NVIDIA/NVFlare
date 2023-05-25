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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender


class LogWriter(FLComponent, ABC):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE):
        super().__init__()
        self.sender = self.load_log_sender(event_type)
        self.engine = None

    def load_log_sender(self, event_type: str = ANALYTIC_EVENT_TYPE) -> AnalyticsSender:
        return AnalyticsSender(event_type, self.get_writer_name())

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.sender.engine = fl_ctx.get_engine()

    @abstractmethod
    def get_writer_name(self) -> LogWriterName:
        pass
