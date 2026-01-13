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

"""Base writer class for Fox tracking in subprocess mode.

This provides a similar interface to nvflare.client.tracking._BaseWriter
but designed for subprocess execution where metrics are sent via CellNet.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType


class BaseWriter(ABC):
    """Abstract base class for tracking writers in Fox subprocess mode.

    Writers are used in the subprocess (FoxWorker) to log metrics during
    training. The metrics are sent via CellNet to FoxExecutor, which then
    forwards them to the configured tracking receiver (MLflow, TensorBoard, etc.).

    Subclasses should implement the `log` method to send metrics appropriately.

    Example usage in training code:
        writer = SubprocessWriter()  # or TensorBoardWriter, MLflowWriter, etc.
        writer.log("loss", 0.5, AnalyticsDataType.SCALAR)
        writer.log("accuracy", 0.95, AnalyticsDataType.SCALAR, step=100)
    """

    def __init__(self):
        """Initialize the writer."""
        # Get rank for DDP - only rank 0 should log
        self.rank = os.environ.get("RANK", "0")
        self.local_rank = os.environ.get("LOCAL_RANK", "0")
        self.site_name = os.environ.get("FOX_SITE_NAME", "unknown")

    def _check_rank(self):
        """Check if this is rank 0 (only rank 0 should log in DDP)."""
        if self.rank != "0":
            raise RuntimeError("Only rank 0 can log metrics in DDP mode!")

    @abstractmethod
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
        """Log a metric.

        Args:
            key: Metric name.
            value: Metric value.
            data_type: Type of the metric (SCALAR, TEXT, IMAGE, etc.).
            step: Optional step number.
            epoch: Optional epoch number.
            global_step: Optional global step number.
            **kwargs: Additional key-value pairs to log.
        """
        pass

    def log_scalar(self, key: str, value: float, step: Optional[int] = None, **kwargs):
        """Log a scalar metric (convenience method)."""
        self.log(key, value, AnalyticsDataType.SCALAR, step=step, **kwargs)

    def log_text(self, key: str, value: str, step: Optional[int] = None, **kwargs):
        """Log a text metric (convenience method)."""
        self.log(key, value, AnalyticsDataType.TEXT, step=step, **kwargs)

    def log_metrics(self, metrics: dict, step: Optional[int] = None, **kwargs):
        """Log multiple scalar metrics at once.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number.
            **kwargs: Additional arguments passed to each log call.
        """
        for key, value in metrics.items():
            self.log_scalar(key, value, step=step, **kwargs)

    def close(self):
        """Close the writer and release resources."""
        pass
