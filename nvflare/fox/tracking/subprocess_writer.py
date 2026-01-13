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

"""Subprocess writer that sends metrics via CellNet to FoxExecutor.

This writer is used inside FoxWorker (subprocess) to send metrics to the
parent FoxExecutor, which then fires events to the tracking receiver.

Architecture:
    FoxWorker (subprocess)
        │
        │ SubprocessWriter.log(key, value, data_type)
        ▼
    CellNet (METRICS_CHANNEL / METRICS_TOPIC)
        │
        ▼
    FoxExecutor
        │
        │ fire_event(TOPIC_LOG_DATA, msg)
        ▼
    MetricsReceiver (MLflow, TensorBoard, etc.)
"""

from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

from .base_writer import BaseWriter
from .constants import METRICS_CHANNEL, METRICS_TOPIC, MetricsKey

# Global writer instance for easy access
_global_writer: Optional["SubprocessWriter"] = None


def get_writer() -> Optional["SubprocessWriter"]:
    """Get the global subprocess writer instance."""
    return _global_writer


def set_writer(writer: "SubprocessWriter"):
    """Set the global subprocess writer instance."""
    global _global_writer
    _global_writer = writer


class SubprocessWriter(BaseWriter):
    """Writer that sends metrics via CellNet to parent FoxExecutor.

    This writer is used in subprocess mode where the training runs in a
    separate process (e.g., via torchrun). Metrics are sent to the parent
    FoxExecutor which forwards them to the configured tracking system.

    Example:
        writer = SubprocessWriter(cell, parent_fqcn)
        writer.log("train/loss", 0.5, AnalyticsDataType.SCALAR, step=100)
        writer.log("train/accuracy", 0.95, AnalyticsDataType.SCALAR, step=100)
    """

    def __init__(
        self,
        cell: CoreCell,
        parent_fqcn: str,
        timeout: float = 5.0,
        fire_and_forget: bool = True,
    ):
        """Initialize SubprocessWriter.

        Args:
            cell: CellNet cell for communication.
            parent_fqcn: FQCN of parent FoxExecutor.
            timeout: Timeout for metric sends (only used if fire_and_forget=False).
            fire_and_forget: If True, don't wait for acknowledgment (faster).
        """
        super().__init__()
        self.cell = cell
        self.parent_fqcn = parent_fqcn
        self.timeout = timeout
        self.fire_and_forget = fire_and_forget
        self.logger = get_obj_logger(self)

        # Register as global writer for easy access
        set_writer(self)

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
        """Log a metric by sending it to parent FoxExecutor via CellNet.

        Args:
            key: Metric name.
            value: Metric value.
            data_type: Type of the metric.
            step: Optional step number.
            epoch: Optional epoch number.
            global_step: Optional global step number.
            **kwargs: Additional key-value pairs.
        """
        # Only rank 0 logs in DDP mode
        self._check_rank()

        # Build metric message
        payload = {
            MetricsKey.KEY: key,
            MetricsKey.VALUE: value,
            MetricsKey.DATA_TYPE: data_type.value if hasattr(data_type, "value") else data_type,
            MetricsKey.SITE_NAME: self.site_name,
            MetricsKey.RANK: self.rank,
        }

        if step is not None:
            payload[MetricsKey.STEP] = step
        if epoch is not None:
            payload[MetricsKey.EPOCH] = epoch
        if global_step is not None:
            payload[MetricsKey.GLOBAL_STEP] = global_step

        # Add any additional kwargs
        payload.update(kwargs)

        # Send via CellNet
        try:
            msg = Message(payload=payload)

            if self.fire_and_forget:
                # Fire and forget - don't wait for response
                self.cell.fire_and_forget(
                    channel=METRICS_CHANNEL,
                    topic=METRICS_TOPIC,
                    target=self.parent_fqcn,
                    message=msg,
                )
            else:
                # Wait for acknowledgment
                self.cell.send_request(
                    channel=METRICS_CHANNEL,
                    topic=METRICS_TOPIC,
                    target=self.parent_fqcn,
                    request=msg,
                    timeout=self.timeout,
                )

            self.logger.debug(f"Sent metric: {key}={value}")

        except Exception as e:
            self.logger.warning(f"Failed to send metric {key}: {e}")

    def close(self):
        """Close the writer."""
        global _global_writer
        if _global_writer is self:
            _global_writer = None


# Convenience functions for logging without needing writer instance
def log(key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
    """Log a metric using the global writer."""
    writer = get_writer()
    if writer:
        writer.log(key, value, data_type, **kwargs)


def log_scalar(key: str, value: float, step: Optional[int] = None, **kwargs):
    """Log a scalar metric using the global writer."""
    writer = get_writer()
    if writer:
        writer.log_scalar(key, value, step=step, **kwargs)


def log_metrics(metrics: dict, step: Optional[int] = None, **kwargs):
    """Log multiple metrics using the global writer."""
    writer = get_writer()
    if writer:
        writer.log_metrics(metrics, step=step, **kwargs)
