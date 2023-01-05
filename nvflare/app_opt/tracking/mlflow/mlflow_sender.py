# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Dict, Optional, Union

import PIL
import numpy

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.tracker_types import TrackConst, TrackerName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender


class MLFlowSender(AnalyticsSender):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE):
        """send tracking data to MLFLOW"""
        super().__init__(event_type=event_type)

    def get_tracker_name(self) -> TrackerName:
        return TrackerName.MLFLOW

    def log_param(self, key: str, value: any) -> None:
        self._add(tag=key, value=value, data_type=AnalyticsDataType.PARAMETER)

    def log_params(self, values: dict) -> None:
        self._add(tag="params", value=values, data_type=AnalyticsDataType.PARAMETERS)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Args:
            key: – Metric name (string). This string may only contain alphanumerics, underscores (_), dashes (-),
                   periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250,
                   but some may support larger keys.
            value: – Metric value (float). Note that some special values such as +/- Infinity may be replaced by other
                    values depending on the store. For example, the SQLAlchemy store replaces +/- Infinity with
                    max / min float values. All backend stores will support values up to length 5000, but some may
                    support larger values.
            step: – Metric step (int). Defaults to zero if unspecified.
        Returns: None
        """
        self._add(tag=key, value=value, data_type=AnalyticsDataType.METRIC, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics for the current run. If no run is active, this method will create a new active run.
        Args:
            metrics: – Dictionary of metric_name: String -> value: Float. Note that some special values such as +/-
                       Infinity may be replaced by other values depending on the store. For example, sql based store
                       may replace +/- Infinity with max / min float values.

            step: – A single integer step at which to log the specified Metrics. If unspecified, each metric is
                    logged at step zero.

        Returns: None
        """
        self._add(tag="metrics", value=metrics, data_type=AnalyticsDataType.METRICS, global_step=step)

    def log_text(self, text: str, artifact_file_path: str):
        """
        Args:
            text – String containing text to log.
            artifact_file_path – The run-relative artifact file path in posixpath format to which the text is saved (e.g. “dir/file.txt”).
        Returns: None
        """
        self._add(tag="text", value=text, data_type=AnalyticsDataType.TEXT, path=artifact_file_path)

    def set_tag(self, key: str, tag: any) -> None:
        self._add(tag=key, value=tag, data_type=AnalyticsDataType.TAG)

    def set_tags(self, key: str, tags: dict) -> None:
        self._add(tag="tags", value=tags, data_type=AnalyticsDataType.TAG)

    #
    # def register_model(model_uri,
    #                    name,
    #                    await_registration_for=300,
    #                    *,
    #                    tags: Optional[Dict[str, Any]] = None) -> ModelVersion:
    #     pass

    def flush(self):
        """Flushes out the message.

        This is doing nothing, it is defined for mimic the PyTorch SummaryWriter behavior.
        """
        pass
