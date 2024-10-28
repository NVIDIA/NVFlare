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


class MLflowWriter(LogWriter):
    def __init__(self, event_type: str = ANALYTIC_EVENT_TYPE):
        """MLflowWriter mimics the usage of mlflow.

        Users can replace the import of mlflow with MLflowWriter. They would then use
        MLflowWriter the same as they would use mlflow. MLflowWriter will send log records to
        the receiver.

        Args:
            event_type (str, optional): _description_. Defaults to ANALYTIC_EVENT_TYPE.
        """
        super().__init__(event_type)

    def get_writer_name(self) -> LogWriterName:
        """Returns "MLFLOW"."""
        return LogWriterName.MLFLOW

    def get_default_metric_data_type(self) -> AnalyticsDataType:
        return AnalyticsDataType.METRICS

    def log_param(self, key: str, value: any) -> None:
        """Log a parameter (e.g. model hyperparameter) under the current run.

        Args:
            key (str): Parameter name. This string may only contain alphanumerics,
                underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores support keys up to length 250, but some may
                support larger keys.
            value (any): Parameter value, of type string, but will be string-ified if not.
                All backend stores support values up to length 500, but some
                may support larger values.
        """
        self.write(tag=key, value=value, data_type=AnalyticsDataType.PARAMETER)

    def log_params(self, values: dict) -> None:
        """Log a batch of params for the current run.

        Args:
            values (dict): Dictionary of param_name: String -> value: (String, but will be string-ified if not)
        """
        self.write(tag="params", value=values, data_type=AnalyticsDataType.PARAMETERS)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric under the current run.

        Args:
            key (str): Metric name. This string may only contain alphanumerics, underscores (_), dashes (-),
                periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250,
                but some may support larger keys.
            value (float): Metric value. Note that some special values such as +/- Infinity may be replaced by other
                values depending on the store. For example, the SQLAlchemy store replaces +/- Infinity with
                max / min float values. All backend stores will support values up to length 5000, but some may
                support larger values.
            step (int, optional): Metric step. Defaults to zero if unspecified.
        """
        self.write(tag=key, value=value, data_type=AnalyticsDataType.METRIC, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics for the current run.

        Args:
            metrics (dict): Dictionary of metric_name: String -> value: Float. Note that some special values such as +/-
                Infinity may be replaced by other values depending on the store. For example, sql based store
                may replace +/- Infinity with max / min float values.
            step (int, optional): A single integer step at which to log the specified Metrics. If unspecified, each metric is
                logged at step zero.
        """
        self.write(tag="metrics", value=metrics, data_type=AnalyticsDataType.METRICS, global_step=step)

    def log_text(self, text: str, artifact_file_path: str) -> None:
        """Log text as an artifact under the current run.

        Args:
            text (str): String of text to log.
            artifact_file_path (str): The run-relative artifact file path in posixpath format
                to which the text is saved (e.g. “dir/file.txt”).
        """
        self.write(tag="text", value=text, data_type=AnalyticsDataType.TEXT, path=artifact_file_path)

    def set_tag(self, key: str, tag: any) -> None:
        """Set a tag under the current run.

        Args:
            key (str): Name of the tag.
            tag (any): Tag value (string, but will be string-ified if not).
                All backend stores will support values up to length 5000, but some
                may support larger values.
        """
        self.write(tag=key, value=tag, data_type=AnalyticsDataType.TAG)

    def set_tags(self, tags: dict) -> None:
        """Log a batch of tags for the current run.

        Args:
            tags (dict): Dictionary of tag_name: String -> value: (String, but will be string-ified if
                not)
        """
        self.write(tag="tags", value=tags, data_type=AnalyticsDataType.TAGS)
