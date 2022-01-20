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

import logging
import os
from logging import LogRecord
from typing import List, Optional

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import LogMessageTag
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class LogAnalyticsReceiver(AnalyticsReceiver):
    CLIENT_LOG_FOLDER = "client_log"

    def __init__(self, events: Optional[List[str]] = None, formatter=None):
        if events is None:
            events = [AppEventType.LOGGING_EVENT_TYPE, f"fed.{AppEventType.LOGGING_EVENT_TYPE}"]
        super().__init__(events=events)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_log_dir = None
        if formatter is None:
            formatter = ""
        self.formatter = formatter

        self.handlers = {}

    def initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_run_number())
        self.root_log_dir = os.path.join(run_dir, LogAnalyticsReceiver.CLIENT_LOG_FOLDER)
        os.makedirs(self.root_log_dir, exist_ok=True)

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        try:
            dxo = from_shareable(shareable)
            analytic_data = AnalyticsData.from_dxo(dxo)
            data_type = analytic_data.data_type

            if data_type == AnalyticsDataType.LOG_RECORD:
                handler = self.handlers.get(record_origin)
                if not handler:
                    handler = self._create_log_handler(record_origin)

                record: LogRecord = dxo.data.get(LogMessageTag.LOG_RECORD)
                handler.emit(record)
        except ValueError:
            self.logger.error(f"Failed to save the log received: {record_origin}")

    def _create_log_handler(self, record_origin):
        filename = os.path.join(self.root_log_dir, record_origin + ".log")
        handler = logging.FileHandler(filename)
        handler.setFormatter(logging.Formatter(self.formatter))
        self.handlers[record_origin] = handler
        return handler

    def finalize(self, fl_ctx: FLContext):
        pass
