# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Metrics handler for receiving subprocess tracking metrics.

This module provides the MetricsHandler that runs in CollabExecutor to receive
metrics from subprocess workers and log them using the standard event mechanism.
"""

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

from .tracking import METRICS_CHANNEL, METRICS_LOG_TOPIC, MetricsKey

# Event topic for logging data (same as used by in-process tracking)
TOPIC_LOG_DATA = "analytix_log_data"


class MetricsHandler:
    """Handler for receiving and logging subprocess metrics.

    This handler is registered in CollabExecutor when using subprocess mode.
    It receives metrics from CollabWorker via CellNet and fires events so
    that the standard tracking receivers (MLflow, TensorBoard, W&B) can
    process them.
    """

    def __init__(self, event_manager, site_name: str):
        """Initialize MetricsHandler.

        Args:
            event_manager: Event manager for firing log events
            site_name: Name of this site
        """
        self.event_manager = event_manager
        self.site_name = site_name
        self.logger = get_obj_logger(self)

    def register(self, cell: CoreCell):
        """Register the metrics handler with CellNet.

        Args:
            cell: CellNet cell to register with
        """
        self.logger.debug(f"Registering metrics handler for {METRICS_CHANNEL}/{METRICS_LOG_TOPIC}")
        cell.register_request_cb(
            channel=METRICS_CHANNEL,
            topic=METRICS_LOG_TOPIC,
            cb=self._handle_metrics,
        )

    def _handle_metrics(self, request: Message) -> Message:
        """Handle incoming metrics from subprocess worker.

        Args:
            request: CellNet message containing metrics

        Returns:
            Acknowledgment message
        """
        try:
            payload = request.payload
            if not isinstance(payload, dict):
                self.logger.warning(f"Invalid metrics payload: {type(payload)}")
                return Message(payload={"status": "error", "message": "invalid payload"})

            key = payload.get(MetricsKey.KEY)
            value = payload.get(MetricsKey.VALUE)
            data_type_value = payload.get(MetricsKey.DATA_TYPE)
            site_name = payload.get(MetricsKey.SITE_NAME, self.site_name)
            kwargs = payload.get(MetricsKey.KWARGS, {})

            # Convert data_type value back to enum if needed
            if isinstance(data_type_value, str):
                try:
                    data_type = AnalyticsDataType(data_type_value)
                except ValueError:
                    data_type = AnalyticsDataType.SCALAR
            elif isinstance(data_type_value, int):
                data_type = AnalyticsDataType(data_type_value)
            else:
                data_type = data_type_value

            self.logger.debug(f"Received metric from {site_name}: {key}={value}")

            # Fire event for tracking receivers to process
            msg = dict(key=key, value=value, data_type=data_type, sender=site_name, **kwargs)
            self.event_manager.fire_event(TOPIC_LOG_DATA, msg)

            return Message(payload={"status": "ok"})

        except Exception as e:
            self.logger.exception(f"Error handling metrics: {e}")
            return Message(payload={"status": "error", "message": str(e)})


def register_metrics_handler(cell: CoreCell, event_manager, site_name: str) -> MetricsHandler:
    """Create and register a metrics handler.

    Args:
        cell: CellNet cell to register with
        event_manager: Event manager for firing log events
        site_name: Name of this site

    Returns:
        The registered MetricsHandler
    """
    handler = MetricsHandler(event_manager, site_name)
    handler.register(cell)
    return handler
