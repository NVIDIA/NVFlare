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
from threading import Lock
from typing import List, Optional

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import LogMessageTag
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.streaming import AnalyticsReceiver, create_analytic_dxo, send_analytic_dxo
from nvflare.widgets.widget import Widget


class LogAnalyticsSender(Widget, logging.StreamHandler):
    def __init__(self, log_level=None):
        """Sends the log record.

        Args:
            log_level: log_level threshold
        """
        Widget.__init__(self)
        logging.StreamHandler.__init__(self)
        if log_level is None:
            log_level = "INFO"

        self.log_level = getattr(logging, log_level)
        if not isinstance(self.log_level, int):
            raise ValueError(f"log_level must be integer. Got: {self.log_level}")

        if self.log_level < logging.INFO:
            raise ValueError(
                f"LogAnalyticsSender log level must be higher than or equal to logging.INFO: {logging.INFO}. "
                f"Got: {self.log_level}"
            )

        self.engine = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.engine = fl_ctx.get_engine()
            logging.root.addHandler(self)
        elif event_type == EventType.END_RUN:
            logging.root.removeHandler(self)

    def emit(self, record: LogRecord):
        """Sends the log record.

        When the log_level higher than the configured level, sends the log record.
        Args:
            record: logging record
        """
        if record.levelno >= self.log_level and self.engine:
            dxo = create_analytic_dxo(
                tag=LogMessageTag.LOG_RECORD, value=record, data_type=AnalyticsDataType.LOG_RECORD
            )
            with self.engine.new_context() as fl_ctx:
                send_analytic_dxo(self, dxo=dxo, fl_ctx=fl_ctx, event_type=AppEventType.LOGGING_EVENT_TYPE)
            self.flush()


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
        self.handlers_lock = Lock()

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
                with self.handlers_lock:
                    handler = self.handlers.get(record_origin)
                    if not handler:
                        handler = self._create_log_handler(record_origin)
                        self.handlers[record_origin] = handler

                record: LogRecord = dxo.data.get(LogMessageTag.LOG_RECORD)
                handler.emit(record)
        except ValueError:
            self.logger.error(f"Failed to save the log received: {record_origin}")

    def _create_log_handler(self, record_origin):
        filename = os.path.join(self.root_log_dir, record_origin + ".log")
        handler = logging.FileHandler(filename)
        handler.setFormatter(logging.Formatter(self.formatter))
        return handler

    def finalize(self, fl_ctx: FLContext):
        pass
