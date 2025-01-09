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
from nvflare.app_common.tracking.tracker_types import LogWriterName

from .api import log


class SummaryWriter:
    """SummaryWriter mimics the usage of Tensorboard's SummaryWriter.

    Users can replace the import of Tensorboard's SummaryWriter with FLARE's SummaryWriter.
    They would then use SummaryWriter the same as before.
    SummaryWriter will send log records to the FLARE system.
    """

    def add_scalar(self, tag: str, scalar: float, global_step: Optional[int] = None, **kwargs):
        """Sends a scalar.

        Args:
            tag (str): Data identifier.
            scalar (float): Value to send.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        log(
            key=tag,
            value=scalar,
            data_type=AnalyticsDataType.SCALAR,
            global_step=global_step,
            writer=LogWriterName.TORCH_TB,
            **kwargs,
        )

    def add_scalars(self, tag: str, scalars: dict, global_step: Optional[int] = None, **kwargs):
        """Sends scalars.

        Args:
            tag (str): The parent name for the tags.
            scalars (dict): Key-value pair storing the tag and corresponding values.
            global_step (optional, int): Global step value.
            **kwargs: Additional arguments to pass to the receiver side.
        """
        log(
            key=tag,
            value=scalars,
            data_type=AnalyticsDataType.SCALARS,
            global_step=global_step,
            writer=LogWriterName.TORCH_TB,
            **kwargs,
        )

    def flush(self):
        """Skip flushing which would normally write the event file to disk"""
        pass


class WandBWriter:
    """WandBWriter mimics the usage of weights and biases.

    Users can replace the import of wandb with FLARE's WandBWriter.
    They would then use WandBWriter the same as they would use wandb.
    WandBWriter will send log records to the FLARE system.
    """

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics for the current run.

        Args:
            metrics (Dict[str, float]): Dictionary of metric_name of type String to Float values.
            step (int, optional): A single integer step at which to log the specified Metrics.
        """
        log(
            key="metrics",
            value=metrics,
            data_type=AnalyticsDataType.METRICS,
            global_step=step,
            writer=LogWriterName.WANDB,
        )


class MLflowWriter:
    """MLflowWriter mimics the usage of MLflow.

    Users can replace the import of MLflow with FLARE's MLflowWriter.
    They would then use MLflowWriter the same as they would use MLflow.
    MLflowWriter will send log records to the FLARE system.
    """

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
        log(key=key, value=value, data_type=AnalyticsDataType.PARAMETER, writer=LogWriterName.MLFLOW)

    def log_params(self, values: dict) -> None:
        """Log a batch of params for the current run.

        Args:
            values (dict): Dictionary of param_name: String -> value: (String, but will be string-ified if not)
        """
        log(key="params", value=values, data_type=AnalyticsDataType.PARAMETERS, writer=LogWriterName.MLFLOW)

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
        log(key=key, value=value, data_type=AnalyticsDataType.METRIC, global_step=step, writer=LogWriterName.MLFLOW)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics for the current run.

        Args:
            metrics (dict): Dictionary of metric_name: String -> value: Float. Note that some special values such as +/-
                Infinity may be replaced by other values depending on the store. For example, sql based store
                may replace +/- Infinity with max / min float values.
            step (int, optional): A single integer step at which to log the specified Metrics. If unspecified, each metric is
                logged at step zero.
        """
        log(
            key="metrics",
            value=metrics,
            data_type=AnalyticsDataType.METRICS,
            global_step=step,
            writer=LogWriterName.MLFLOW,
        )

    def log_text(self, text: str, artifact_file_path: str) -> None:
        """Log text as an artifact under the current run.

        Args:
            text (str): String of text to log.
            artifact_file_path (str): The run-relative artifact file path in posixpath format
                to which the text is saved (e.g. “dir/file.txt”).
        """
        log(
            key="text",
            value=text,
            data_type=AnalyticsDataType.TEXT,
            path=artifact_file_path,
            writer=LogWriterName.MLFLOW,
        )

    def set_tag(self, key: str, tag: any) -> None:
        """Set a tag under the current run.

        Args:
            key (str): Name of the tag.
            tag (any): Tag value (string, but will be string-ified if not).
                All backend stores will support values up to length 5000, but some
                may support larger values.
        """
        log(key=key, value=tag, data_type=AnalyticsDataType.TAG, writer=LogWriterName.MLFLOW)

    def set_tags(self, tags: dict) -> None:
        """Log a batch of tags for the current run.

        Args:
            tags (dict): Dictionary of tag_name: String -> value: (String, but will be string-ified if
                not)
        """
        log(key="tags", value=tags, data_type=AnalyticsDataType.TAGS, writer=LogWriterName.MLFLOW)
