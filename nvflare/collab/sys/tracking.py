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

"""Subprocess tracking writers for experiment metrics.

This module provides tracking writers that work in subprocess mode by sending
metrics over CellNet to the parent CollabExecutor, which then logs them using
the standard event mechanism.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Subprocess (Worker)                                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │ User's training code                                            │    │
    │  │   writer.log("loss", 0.5, AnalyticsDataType.SCALAR)             │    │
    │  └────────────────────────────────┬────────────────────────────────┘    │
    │                                   │                                     │
    │                                   ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │ SubprocessTrackingWriter                                        │    │
    │  │   → Sends metrics via CellNet METRICS_CHANNEL                   │    │
    │  └────────────────────────────────┬────────────────────────────────┘    │
    └───────────────────────────────────┼─────────────────────────────────────┘
                                        │ CellNet
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ CollabExecutor                                                               │
    │  ┌─────────────────────────────────────────────────────────────────────┐  │
    │  │ MetricsHandler                                                      │  │
    │  │   → Receives metrics, fires event via event_manager                 │  │
    │  │   → event_manager.fire_event(TOPIC_LOG_DATA, msg)                   │  │
    │  └─────────────────────────────────────────────────────────────────────┘  │
    │                                        │                                  │
    │                                        ▼                                  │
    │  ┌─────────────────────────────────────────────────────────────────────┐  │
    │  │ MLflow/TensorBoard/W&B Receiver (on server via add_experiment_tracking)│
    │  └─────────────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────────────────┘
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

# CellNet channel and topic for metrics
METRICS_CHANNEL = "collab_metrics"
METRICS_LOG_TOPIC = "log"


# Keys for metrics message payload
class MetricsKey:
    KEY = "key"
    VALUE = "value"
    DATA_TYPE = "data_type"
    GLOBAL_STEP = "global_step"
    KWARGS = "kwargs"
    SITE_NAME = "site_name"
    RANK = "rank"


class BaseSubprocessWriter(ABC):
    """Base class for subprocess tracking writers.

    These writers send metrics over CellNet to the parent CollabExecutor
    instead of logging locally. This enables experiment tracking when
    training runs in a subprocess (e.g., with torchrun).
    """

    def __init__(self):
        self.logger = get_obj_logger(self)
        self._cell: Optional[CoreCell] = None
        self._parent_fqcn: Optional[str] = None
        self._site_name: Optional[str] = None
        self._rank: str = os.environ.get("RANK", "0")
        self._timeout: float = 10.0

    def setup(self, cell: CoreCell, parent_fqcn: str, site_name: str, timeout: float = 10.0):
        """Set up the writer with CellNet connection info.

        Args:
            cell: CellNet cell for communication
            parent_fqcn: FQCN of parent CollabExecutor
            site_name: Name of this site
            timeout: Timeout for sending metrics
        """
        self._cell = cell
        self._parent_fqcn = parent_fqcn
        self._site_name = site_name
        self._timeout = timeout

    def _send_metrics(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Send metrics to parent CollabExecutor via CellNet.

        Args:
            key: Metric key/name
            value: Metric value
            data_type: Type of analytics data
            **kwargs: Additional arguments (e.g., global_step)
        """
        if self._rank != "0":
            # Only rank 0 sends metrics in DDP mode
            return

        if not self._cell or not self._parent_fqcn:
            self.logger.warning("Writer not set up. Call setup() first.")
            return

        payload = {
            MetricsKey.KEY: key,
            MetricsKey.VALUE: value,
            MetricsKey.DATA_TYPE: data_type.value if hasattr(data_type, "value") else data_type,
            MetricsKey.SITE_NAME: self._site_name,
            MetricsKey.RANK: self._rank,
            MetricsKey.KWARGS: kwargs,
        }

        try:
            self._cell.fire_and_forget(
                channel=METRICS_CHANNEL,
                topic=METRICS_LOG_TOPIC,
                target=self._parent_fqcn,
                message=Message(payload=payload),
            )
        except Exception as e:
            self.logger.warning(f"Failed to send metrics: {e}")

    @abstractmethod
    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric.

        Args:
            key: Metric key/name
            value: Metric value
            data_type: Type of analytics data
            **kwargs: Additional arguments
        """
        pass


class SubprocessMLflowWriter(BaseSubprocessWriter):
    """MLflow-style writer for subprocess tracking.

    Sends metrics to parent CollabExecutor which then logs to MLflow.
    """

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric to MLflow via parent executor.

        Args:
            key: Metric name
            value: Metric value
            data_type: Type of analytics data
            **kwargs: Additional arguments (e.g., step)
        """
        self._send_metrics(key, value, data_type, **kwargs)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a scalar metric (MLflow-style API).

        Args:
            key: Metric name
            value: Metric value
            step: Training step
        """
        kwargs = {}
        if step is not None:
            kwargs["global_step"] = step
        self.log(key, value, AnalyticsDataType.SCALAR, **kwargs)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log multiple metrics (MLflow-style API).

        Args:
            metrics: Dict of metric name to value
            step: Training step
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)


class SubprocessTensorBoardWriter(BaseSubprocessWriter):
    """TensorBoard-style writer for subprocess tracking.

    Sends metrics to parent CollabExecutor which then logs to TensorBoard.
    """

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric to TensorBoard via parent executor.

        Args:
            key: Metric tag
            value: Metric value
            data_type: Type of analytics data
            **kwargs: Additional arguments (e.g., global_step)
        """
        self._send_metrics(key, value, data_type, **kwargs)

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """Add a scalar value (TensorBoard-style API).

        Args:
            tag: Data identifier
            scalar_value: Value to save
            global_step: Global step value
        """
        kwargs = {}
        if global_step is not None:
            kwargs["global_step"] = global_step
        self.log(tag, scalar_value, AnalyticsDataType.SCALAR, **kwargs)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: Optional[int] = None):
        """Add multiple scalars (TensorBoard-style API).

        Args:
            main_tag: Parent name for the tags
            tag_scalar_dict: Dict of tag to scalar value
            global_step: Global step value
        """
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.add_scalar(full_tag, value, global_step)


class SubprocessWandBWriter(BaseSubprocessWriter):
    """Weights & Biases style writer for subprocess tracking.

    Sends metrics to parent CollabExecutor which then logs to W&B.
    """

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric to W&B via parent executor.

        Args:
            key: Metric name
            value: Metric value
            data_type: Type of analytics data
            **kwargs: Additional arguments
        """
        self._send_metrics(key, value, data_type, **kwargs)

    def log_dict(self, data: dict, step: Optional[int] = None):
        """Log a dictionary of values (W&B-style API).

        Args:
            data: Dict of metric names to values
            step: Training step
        """
        kwargs = {}
        if step is not None:
            kwargs["global_step"] = step
        self.log("metrics", data, AnalyticsDataType.METRICS, **kwargs)


def get_subprocess_writer(tracking_type: str) -> BaseSubprocessWriter:
    """Get the appropriate subprocess writer for the tracking type.

    Args:
        tracking_type: Type of tracking ("mlflow", "tensorboard", "wandb")

    Returns:
        Appropriate subprocess writer instance
    """
    writers = {
        "mlflow": SubprocessMLflowWriter,
        "tensorboard": SubprocessTensorBoardWriter,
        "wandb": SubprocessWandBWriter,
    }

    if tracking_type not in writers:
        raise ValueError(f"Unknown tracking type: {tracking_type}. Supported: {list(writers.keys())}")

    return writers[tracking_type]()
