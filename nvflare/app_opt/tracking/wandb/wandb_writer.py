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

from typing import Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.tracking.log_writer import LogWriter
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE


class WandBWriter(LogWriter):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE):
        super().__init__(event_type)

    def get_writer_name(self) -> LogWriterName:
        """Returns "WEIGHTS_AND_BIASES"."""
        return LogWriterName.WANDB

    def get_default_metric_data_type(self) -> AnalyticsDataType:
        return AnalyticsDataType.METRICS

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics for the current run.

        Args:
            metrics (Dict[str, float]): Dictionary of metric_name of type String to Float values.
            step (int, optional): A single integer step at which to log the specified Metrics.
        """
        self.write(tag="metrics", value=metrics, data_type=AnalyticsDataType.METRICS, global_step=step)
