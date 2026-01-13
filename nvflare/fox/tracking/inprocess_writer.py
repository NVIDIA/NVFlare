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

"""In-process writer that fires events directly to the event manager.

This writer is used when training runs in-process (inprocess=True) within
FoxExecutor. It fires log events directly without needing CellNet relay.

Architecture (In-Process Mode):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ FoxExecutor (same process)                                              │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │ User's training code                                            │    │
    │  │   writer.add_scalar("loss", 0.5, global_step=100)               │    │
    │  └────────────────────────────────┬────────────────────────────────┘    │
    │                                   │                                     │
    │                                   ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │ InProcessWriter                                                 │    │
    │  │   → Fires event_manager.fire_event(TOPIC_LOG_DATA, msg)         │    │
    │  └────────────────────────────────┬────────────────────────────────┘    │
    │                                   │                                     │
    │                                   ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │ TrackingReceiver (TensorBoard/MLflow/W&B)                       │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.utils.log_utils import get_obj_logger

from .base_writer import BaseWriter

# Topic for log data events (same as used by tracking system)
TOPIC_LOG_DATA = "analytics_log_data"

# Global in-process writer instance
_inprocess_writer: Optional["InProcessWriter"] = None


def get_inprocess_writer() -> Optional["InProcessWriter"]:
    """Get the global in-process writer instance."""
    return _inprocess_writer


def set_inprocess_writer(writer: "InProcessWriter"):
    """Set the global in-process writer instance."""
    global _inprocess_writer
    _inprocess_writer = writer


class InProcessWriter(BaseWriter):
    """Writer that fires events directly to the event manager.

    This writer is used in in-process mode where training code runs
    in the same process as FoxExecutor. It fires log events directly
    to the event manager without needing CellNet relay.

    Example:
        writer = InProcessWriter(event_manager)
        writer.log("train/loss", 0.5, AnalyticsDataType.SCALAR, step=100)
    """

    def __init__(self, event_manager, site_name: str = "local"):
        """Initialize InProcessWriter.

        Args:
            event_manager: Event manager to fire log events.
            site_name: Name of the site (for multi-client tracking).
        """
        super().__init__()
        self.event_manager = event_manager
        self.site_name = site_name
        self.logger = get_obj_logger(self)

        # Register as global writer
        set_inprocess_writer(self)

    def log(
        self,
        key: str,
        value: Any,
        data_type: AnalyticsDataType,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ):
        """Log a metric by firing an event directly.

        Args:
            key: Metric name.
            value: Metric value.
            data_type: Type of the metric.
            step: Optional step number.
            epoch: Optional epoch number.
            global_step: Optional global step number.
            **kwargs: Additional key-value pairs.
        """
        # Build event message
        msg = {
            "key": key,
            "value": value,
            "data_type": data_type,
            "sender": self.site_name,
        }

        if step is not None:
            msg["step"] = step
        if epoch is not None:
            msg["epoch"] = epoch
        if global_step is not None:
            msg["global_step"] = global_step

        # Add any additional kwargs
        msg.update(kwargs)

        # Fire event directly
        try:
            self.event_manager.fire_event(TOPIC_LOG_DATA, msg)
            self.logger.debug(f"Logged metric: {key}={value}")
        except Exception as e:
            self.logger.warning(f"Failed to log metric {key}: {e}")

    def close(self):
        """Close the writer."""
        global _inprocess_writer
        if _inprocess_writer is self:
            _inprocess_writer = None
