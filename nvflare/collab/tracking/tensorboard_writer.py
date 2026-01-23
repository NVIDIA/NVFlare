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

"""TensorBoard-compatible writer for Collab subprocess mode.

This writer mimics the torch.utils.tensorboard.SummaryWriter API,
allowing users to change only the import statement when running
in Fox subprocess mode.

Original TensorBoard usage:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_scalar("loss", 0.5, global_step=100)

Fox subprocess usage (same API, different import):
    from nvflare.collab.tracking import TensorBoardWriter
    writer = TensorBoardWriter()
    writer.add_scalar("loss", 0.5, global_step=100)
"""

from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType

from .auto_writer import AutoWriter
from .auto_writer import get_writer as get_auto_writer
from .base_writer import BaseWriter

# Global TensorBoard writer instance
_tb_writer: Optional["TensorBoardWriter"] = None


def get_tensorboard_writer() -> Optional["TensorBoardWriter"]:
    """Get the global TensorBoard writer instance."""
    return _tb_writer


class TensorBoardWriter(BaseWriter):
    """TensorBoard-compatible writer for Collab subprocess mode.

    This class provides the same API as torch.utils.tensorboard.SummaryWriter,
    but sends metrics via CellNet to the parent CollabExecutor for logging.

    Example:
        writer = TensorBoardWriter()
        writer.add_scalar("train/loss", loss, global_step=step)
        writer.add_scalars("losses", {"train": 0.5, "val": 0.6}, global_step=step)
    """

    def __init__(self, log_dir: str = None, **kwargs):
        """Initialize TensorBoardWriter.

        Args:
            log_dir: Ignored in Fox mode (compatibility with SummaryWriter).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        super().__init__()
        self._delegate: Optional[AutoWriter] = None
        self._log_dir = log_dir

        # Register as global writer
        global _tb_writer
        _tb_writer = self

    def _get_delegate(self) -> Optional[AutoWriter]:
        """Get the auto writer (lazy initialization)."""
        if self._delegate is None:
            self._delegate = get_auto_writer()
        return self._delegate

    def log(
        self,
        key: str,
        value: Any,
        data_type: AnalyticsDataType,
        step: Optional[int] = None,
        **kwargs,
    ):
        """Log a metric (internal method)."""
        # Skip rank check in in-process mode (no DDP)
        delegate = self._get_delegate()
        if delegate:
            delegate.log(key, value, data_type, step=step, **kwargs)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        new_style: bool = False,
        double_precision: bool = False,
    ):
        """Add scalar data to summary.

        Args:
            tag: Data identifier.
            scalar_value: Value to save.
            global_step: Global step value to record.
            walltime: Optional override for default walltime.
            new_style: Whether to use new style (ignored).
            double_precision: Whether to use double precision (ignored).
        """
        self.log(tag, scalar_value, AnalyticsDataType.SCALAR, step=global_step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        """Add multiple scalars to summary.

        Args:
            main_tag: Parent name for the tags.
            tag_scalar_dict: Dictionary of tag-value pairs.
            global_step: Global step value to record.
            walltime: Optional override for default walltime.
        """
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.log(full_tag, value, AnalyticsDataType.SCALAR, step=global_step)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: str = "tensorflow",
        walltime: Optional[float] = None,
        max_bins: Optional[int] = None,
    ):
        """Add histogram to summary.

        Note: Histogram data is logged as METRICS type and may be
        processed differently by the tracking receiver.
        """
        # Convert to list if tensor
        if hasattr(values, "tolist"):
            values = values.tolist()
        elif hasattr(values, "numpy"):
            values = values.numpy().tolist()
        self.log(tag, values, AnalyticsDataType.METRICS, step=global_step)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
        dataformats: str = "CHW",
    ):
        """Add image data to summary.

        Note: Image data is serialized and sent to the tracking receiver.
        Large images may impact performance.
        """
        self.log(tag, img_tensor, AnalyticsDataType.IMAGE, step=global_step)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        """Add text data to summary."""
        self.log(tag, text_string, AnalyticsDataType.TEXT, step=global_step)

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, Any],
        hparam_domain_discrete: Optional[Dict[str, list]] = None,
        run_name: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        """Add hyperparameters to summary."""
        # Log hyperparameters
        for key, value in hparam_dict.items():
            self.log(f"hparams/{key}", value, AnalyticsDataType.PARAMETER)
        # Log metrics
        for key, value in metric_dict.items():
            self.log(f"hparams/metrics/{key}", value, AnalyticsDataType.SCALAR)

    def flush(self):
        """Flush pending events (no-op in subprocess mode)."""
        pass

    def close(self):
        """Close the writer."""
        global _tb_writer
        if _tb_writer is self:
            _tb_writer = None


# Alias for compatibility
SummaryWriter = TensorBoardWriter
