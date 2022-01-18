import json
import logging
from typing import List, Optional
from logging import LogRecord

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable, DXO
from nvflare.apis.fl_constant import LogMessageTag
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


def _get_sender_log_info(dxo: DXO, record_origin: str) -> str:
    record: LogRecord = dxo.data.get(LogMessageTag.LOG_RECORD)
    data = {"from-client": record_origin,
            "name": record.name,
            "log_level": record.levelname,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "func": record.funcName,
            "msg": record.msg
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
