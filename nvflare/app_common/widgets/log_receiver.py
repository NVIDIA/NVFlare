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

import json
import logging
from logging import LogRecord
from typing import List, Optional

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_constant import LogMessageTag
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


def _get_sender_log_info(dxo: DXO, record_origin: str) -> str:
    record: LogRecord = dxo.data.get(LogMessageTag.LOG_RECORD)
    data = {
        "from-client": record_origin,
        "name": record.name,
        "log_level": record.levelname,
        "pathname": record.pathname,
        "lineno": record.lineno,
        "func": record.funcName,
        "msg": record.msg,
    }
    return json.dumps(data)


class LogAnalyticsReceiver(AnalyticsReceiver):
    def __init__(self, events: Optional[List[str]] = None):
        super().__init__(events=events)
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize(self, fl_ctx: FLContext):
        pass

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        try:
            dxo = from_shareable(shareable)
            analytic_data = AnalyticsData.from_dxo(dxo)
            data_type = analytic_data.data_type

            if data_type == AnalyticsDataType.LOG_RECORD:
                self.logger.info(_get_sender_log_info(dxo, record_origin))
        except ValueError:
            self.logger.error(f"Failed to save the log received: {record_origin}")

    def finalize(self, fl_ctx: FLContext):
        pass
