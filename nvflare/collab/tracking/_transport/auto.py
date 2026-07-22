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

"""Writer that delegates to the Collab writer in the current site process.

Users can use the compatibility writer without depending on its event transport.
"""

from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.fuel.utils.log_utils import get_obj_logger

from .base import BaseWriter

# Global auto writer instance
_auto_writer: Optional["AutoWriter"] = None


def get_auto_writer() -> Optional["AutoWriter"]:
    """Get the global auto writer instance."""
    return _auto_writer


def set_auto_writer(writer: "AutoWriter"):
    """Set the global auto writer instance."""
    global _auto_writer
    _auto_writer = writer


class AutoWriter(BaseWriter):
    """Writer that delegates to the site process's event writer.

    Example:
        from nvflare.collab.tracking import AutoWriter
        writer = AutoWriter()
        writer.log("loss", 0.5, AnalyticsDataType.SCALAR, step=100)
    """

    def __init__(self):
        """Initialize AutoWriter."""
        super().__init__()
        self.logger = get_obj_logger(self)
        self._delegate: Optional[BaseWriter] = None
        self._initialized = False

        # Register as global writer
        set_auto_writer(self)

    def _ensure_initialized(self):
        """Lazily initialize the site-local delegate writer."""
        if self._initialized:
            return

        from .in_process import get_inprocess_writer

        self._delegate = get_inprocess_writer()
        if self._delegate:
            self.logger.debug("AutoWriter using InProcessWriter")
        else:
            self.logger.debug("InProcessWriter not available - metrics may not be logged")

        self._initialized = True

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
        """Log a metric using the appropriate writer.

        Args:
            key: Metric name.
            value: Metric value.
            data_type: Type of the metric.
            step: Optional step number.
            epoch: Optional epoch number.
            global_step: Optional global step number.
            **kwargs: Additional key-value pairs.
        """
        self._ensure_initialized()

        if self._delegate:
            self._delegate.log(key, value, data_type, step=step, epoch=epoch, global_step=global_step, **kwargs)
        else:
            # Silently ignore if no writer available
            # This allows training code to run without tracking configured
            pass

    def close(self):
        """Close the writer."""
        global _auto_writer
        if self._delegate:
            self._delegate.close()
        if _auto_writer is self:
            _auto_writer = None


# Convenience function to get or create an auto writer
def get_writer() -> AutoWriter:
    """Get or create the global auto writer.

    This is the recommended way to get a writer in user training code.
    It automatically detects the execution mode and returns the appropriate writer.

    Example:
        from nvflare.collab.tracking._transport.auto import get_writer
        writer = get_writer()
        writer.log_scalar("loss", 0.5, step=100)
    """
    global _auto_writer
    if _auto_writer is None:
        _auto_writer = AutoWriter()
    return _auto_writer
