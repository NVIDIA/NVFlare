# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.tracking.log_writer_me import LogWriterForMetricsExchanger
from nvflare.app_common.tracking.tracker_types import LogWriterName


class TBWriterForMetricsExchanger(LogWriterForMetricsExchanger):
    """Sends experiment tracking data through MetricsExchanger."""

    def get_writer_name(self) -> LogWriterName:
        return LogWriterName.TORCH_TB

    def add_scalar(self, tag: str, scalar: float, global_step: Optional[int] = None, **kwargs):
        """Sends a scalar.

        Args:
            tag (str): Data identifier.
            scalar (float): Value to send.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self.send_log(key=tag, value=scalar, data_type=AnalyticsDataType.SCALAR, global_step=global_step, **kwargs)

    def add_scalars(self, tag: str, scalars: dict, global_step: Optional[int] = None, **kwargs):
        """Sends scalars.

        Args:
            tag (str): The parent name for the tags.
            scalars (dict): Key-value pair storing the tag and corresponding values.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        self.send_log(key=tag, value=scalars, data_type=AnalyticsDataType.SCALARS, global_step=global_step, **kwargs)

    def flush(self):
        """Flushes out the message.

        This does nothing, it is defined to mimic the PyTorch SummaryWriter behavior.
        """
        pass
