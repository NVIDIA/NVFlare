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

"""MLflow-compatible writer for Fox subprocess mode.

This writer mimics the MLflow logging API, allowing users to change
only the import statement when running in Fox subprocess mode.

Original MLflow usage:
    import mlflow
    mlflow.log_metric("loss", 0.5, step=100)
    mlflow.log_param("learning_rate", 0.001)

Fox subprocess usage (same API, different import):
    from nvflare.fox.tracking import mlflow
    mlflow.log_metric("loss", 0.5, step=100)
    mlflow.log_param("learning_rate", 0.001)
"""

from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType

from .auto_writer import AutoWriter
from .auto_writer import get_writer as get_auto_writer

# Global MLflow writer instance
_mlflow_writer: Optional["MLflowWriter"] = None


def _get_or_create_writer() -> "MLflowWriter":
    """Get or create the global MLflow writer."""
    global _mlflow_writer
    if _mlflow_writer is None:
        _mlflow_writer = MLflowWriter()
    return _mlflow_writer


class MLflowWriter:
    """MLflow-compatible writer for Fox (works in both in-process and subprocess modes).

    This class provides the same API as MLflow's logging functions.
    It automatically detects the execution mode and uses the appropriate
    underlying writer.

    Note: This class does not extend BaseWriter because MLflow API methods
    have different signatures (e.g., log_text takes artifact_file instead of key).
    """

    def __init__(self):
        """Initialize MLflowWriter."""
        self._delegate: Optional[AutoWriter] = None
        self._run_id: Optional[str] = None

        # Register as global writer
        global _mlflow_writer
        _mlflow_writer = self

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
        delegate = self._get_delegate()
        if delegate:
            delegate.log(key, value, data_type, step=step, **kwargs)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Log a metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: A single integer step at which to log the metric.
            timestamp: Time when this metric was calculated (ignored).
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        self.log(key, value, AnalyticsDataType.SCALAR, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metric name to value.
            step: A single integer step at which to log the metrics.
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        for key, value in metrics.items():
            self.log(key, value, AnalyticsDataType.SCALAR, step=step)

    def log_param(
        self,
        key: str,
        value: Any,
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Log a parameter.

        Args:
            key: Parameter name.
            value: Parameter value (will be converted to string).
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        self.log(key, value, AnalyticsDataType.PARAMETER)

    def log_params(
        self,
        params: Dict[str, Any],
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Log multiple parameters.

        Args:
            params: Dictionary of parameter name to value.
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        for key, value in params.items():
            self.log(key, value, AnalyticsDataType.PARAMETER)

    def log_text(
        self,
        text: str,
        artifact_file: str,
        run_id: Optional[str] = None,
    ):
        """Log text as an artifact.

        Args:
            text: Text content.
            artifact_file: Artifact file name/path.
            run_id: Run ID (ignored in subprocess mode).
        """
        self.log(artifact_file, text, AnalyticsDataType.TEXT)

    def set_tag(
        self,
        key: str,
        value: Any,
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Set a tag.

        Args:
            key: Tag name.
            value: Tag value.
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        self.log(f"tag/{key}", value, AnalyticsDataType.TAG)

    def set_tags(
        self,
        tags: Dict[str, Any],
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Set multiple tags.

        Args:
            tags: Dictionary of tag name to value.
            synchronous: Whether to block until logged (ignored).
            run_id: Run ID (ignored in subprocess mode).
        """
        for key, value in tags.items():
            self.log(f"tag/{key}", value, AnalyticsDataType.TAG)


# Module-level functions to mimic mlflow API
def log_metric(
    key: str,
    value: float,
    step: Optional[int] = None,
    timestamp: Optional[int] = None,
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Log a metric (module-level function)."""
    _get_or_create_writer().log_metric(key, value, step, timestamp, synchronous, run_id)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Log multiple metrics (module-level function)."""
    _get_or_create_writer().log_metrics(metrics, step, synchronous, run_id)


def log_param(
    key: str,
    value: Any,
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Log a parameter (module-level function)."""
    _get_or_create_writer().log_param(key, value, synchronous, run_id)


def log_params(
    params: Dict[str, Any],
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Log multiple parameters (module-level function)."""
    _get_or_create_writer().log_params(params, synchronous, run_id)


def log_text(
    text: str,
    artifact_file: str,
    run_id: Optional[str] = None,
):
    """Log text as an artifact (module-level function)."""
    _get_or_create_writer().log_text(text, artifact_file, run_id)


def set_tag(
    key: str,
    value: Any,
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Set a tag (module-level function)."""
    _get_or_create_writer().set_tag(key, value, synchronous, run_id)


def set_tags(
    tags: Dict[str, Any],
    synchronous: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """Set multiple tags (module-level function)."""
    _get_or_create_writer().set_tags(tags, synchronous, run_id)
