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

"""Metrics handler for FoxExecutor to receive metrics from subprocess workers.

This module provides functions to set up metrics reception on the FoxExecutor
side. When metrics are received from a FoxWorker subprocess, they are forwarded
to the event system for tracking receivers to handle.
"""

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

from .constants import METRICS_CHANNEL, METRICS_TOPIC, MetricsKey

# Topic for log data events (same as used by tracking system)
TOPIC_LOG_DATA = "analytics_log_data"


def setup_metrics_handler(
    cell: CoreCell,
    event_manager,
    logger=None,
):
    """Set up metrics handler on FoxExecutor to receive metrics from subprocess.

    Args:
        cell: CellNet cell for receiving metrics.
        event_manager: Event manager to fire log events.
        logger: Optional logger.
    """
    if logger is None:
        logger = get_obj_logger(setup_metrics_handler)

    def handle_metrics(request: Message) -> Message:
        """Handle incoming metrics from subprocess worker."""
        try:
            payload = request.payload
            if not isinstance(payload, dict):
                logger.error(f"Invalid metrics payload: {type(payload)}")
                return new_cell_message(
                    headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION},
                    payload={"error": "Invalid payload"},
                )

            # Extract metric info
            key = payload.get(MetricsKey.KEY)
            value = payload.get(MetricsKey.VALUE)
            data_type = payload.get(MetricsKey.DATA_TYPE)
            site_name = payload.get(MetricsKey.SITE_NAME, "unknown")

            logger.debug(f"Received metric from {site_name}: {key}={value}")

            # Convert data_type back to enum if needed
            if isinstance(data_type, str):
                try:
                    data_type = AnalyticsDataType(data_type)
                except ValueError:
                    data_type = AnalyticsDataType.SCALAR

            # Build event message (similar to how tracking writers do it)
            msg = {
                "key": key,
                "value": value,
                "data_type": data_type,
            }

            # Add optional fields
            if MetricsKey.STEP in payload:
                msg["step"] = payload[MetricsKey.STEP]
            if MetricsKey.EPOCH in payload:
                msg["epoch"] = payload[MetricsKey.EPOCH]
            if MetricsKey.GLOBAL_STEP in payload:
                msg["global_step"] = payload[MetricsKey.GLOBAL_STEP]

            # Add site name for multi-client tracking
            msg["sender"] = site_name

            # Fire event to tracking receiver
            event_manager.fire_event(TOPIC_LOG_DATA, msg)

            return new_cell_message(
                headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK},
                payload={"status": "ok"},
            )

        except Exception as e:
            logger.exception(f"Error handling metrics: {e}")
            return new_cell_message(
                headers={MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION},
                payload={"error": str(e)},
            )

    # Register the handler
    cell.register_request_cb(
        channel=METRICS_CHANNEL,
        topic=METRICS_TOPIC,
        cb=handle_metrics,
    )
    logger.debug(f"Registered metrics handler for {METRICS_CHANNEL}/{METRICS_TOPIC}")


class MetricsRelay:
    """Relay metrics from subprocess to tracking system.

    This class provides a cleaner interface for setting up metrics relay
    in FoxExecutor. It handles both receiving metrics via CellNet and
    forwarding them to the event system.

    Usage in FoxExecutor:
        relay = MetricsRelay(cell, event_manager)
        relay.start()
        # ... run training ...
        relay.stop()
    """

    def __init__(
        self,
        cell: CoreCell,
        event_manager,
        logger=None,
    ):
        """Initialize MetricsRelay.

        Args:
            cell: CellNet cell for receiving metrics.
            event_manager: Event manager for firing log events.
            logger: Optional logger.
        """
        self.cell = cell
        self.event_manager = event_manager
        self.logger = logger or get_obj_logger(self)
        self._started = False

    def start(self):
        """Start the metrics relay."""
        if self._started:
            return

        setup_metrics_handler(self.cell, self.event_manager, self.logger)
        self._started = True
        self.logger.info("Metrics relay started")

    def stop(self):
        """Stop the metrics relay."""
        if not self._started:
            return

        # Note: CellNet doesn't have unregister_request_cb, so we just mark as stopped
        self._started = False
        self.logger.info("Metrics relay stopped")
