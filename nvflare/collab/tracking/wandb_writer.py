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

"""Weights & Biases (wandb) compatible writer for Collab subprocess mode.

This writer mimics the wandb logging API, allowing users to change
only the import statement when running in Fox subprocess mode.

Original W&B usage:
    import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5, "accuracy": 0.9})

Fox subprocess usage (same API, different import):
    from nvflare.collab.tracking import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5, "accuracy": 0.9})
"""

from typing import Any, Dict, Optional, Sequence, Union

from nvflare.apis.analytix import AnalyticsDataType

from .auto_writer import AutoWriter
from .auto_writer import get_writer as get_auto_writer

# Global wandb writer instance
_wandb_writer: Optional["WandbWriter"] = None


def _get_or_create_writer() -> "WandbWriter":
    """Get or create the global wandb writer."""
    global _wandb_writer
    if _wandb_writer is None:
        _wandb_writer = WandbWriter()
    return _wandb_writer


class WandbWriter:
    """Weights & Biases compatible writer for Collab (works in both modes).

    This class provides the same API as wandb's logging functions.
    It automatically detects the execution mode and uses the appropriate
    underlying writer.

    Note: This class does not extend BaseWriter because wandb.log() has
    a different signature (takes a dict instead of key/value/data_type).
    """

    def __init__(self):
        """Initialize WandbWriter."""
        self._delegate: Optional[AutoWriter] = None
        self._step: int = 0
        self._config: Dict[str, Any] = {}

        # Register as global writer
        global _wandb_writer
        _wandb_writer = self

    def _get_delegate(self) -> Optional[AutoWriter]:
        """Get the auto writer (lazy initialization)."""
        if self._delegate is None:
            self._delegate = get_auto_writer()
        return self._delegate

    def _log_internal(
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

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        """Log metrics, media, or custom data.

        Args:
            data: Dictionary of metrics to log.
            step: Step to associate with the log (optional).
            commit: Whether to commit the step (ignored in subprocess mode).
            sync: Whether to sync immediately (ignored in subprocess mode).
        """
        if step is None:
            step = self._step
            if commit is not False:
                self._step += 1

        for key, value in data.items():
            # Determine data type
            if isinstance(value, (int, float)):
                data_type = AnalyticsDataType.SCALAR
            elif isinstance(value, str):
                data_type = AnalyticsDataType.TEXT
            elif isinstance(value, dict):
                data_type = AnalyticsDataType.METRICS
            else:
                data_type = AnalyticsDataType.SCALAR

            self._log_internal(key, value, data_type, step=step)

    def define_metric(
        self,
        name: str,
        step_metric: Optional[str] = None,
        step_sync: Optional[bool] = None,
        hidden: Optional[bool] = None,
        summary: Optional[str] = None,
        goal: Optional[str] = None,
        overwrite: Optional[bool] = None,
    ):
        """Define a custom metric (no-op in subprocess mode).

        Metric definitions are handled by the tracking receiver.
        """
        pass

    def summary_update(self, summary: Dict[str, Any]):
        """Update the run summary."""
        for key, value in summary.items():
            self._log_internal(f"summary/{key}", value, AnalyticsDataType.SCALAR)


class Config:
    """Mock wandb.config object."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def update(self, data: Dict[str, Any]):
        """Update configuration."""
        self._data.update(data)
        # Log config updates
        writer = _get_or_create_writer()
        for key, value in data.items():
            writer._log_internal(f"config/{key}", value, AnalyticsDataType.PARAMETER)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value
            writer = _get_or_create_writer()
            writer._log_internal(f"config/{name}", value, AnalyticsDataType.PARAMETER)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)


# Global config object
config = Config()


# Module-level functions to mimic wandb API
def init(
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    dir: Optional[str] = None,
    resume: Optional[Union[bool, str]] = None,
    reinit: Optional[bool] = None,
    mode: Optional[str] = None,
    **kwargs,
) -> "WandbWriter":
    """Initialize a wandb run (subprocess mode).

    In subprocess mode, this creates the writer but doesn't actually
    initialize a wandb run. The tracking receiver on the server side
    handles the actual wandb initialization.

    Args:
        project: Project name (logged as parameter).
        entity: Entity/team name (logged as parameter).
        config: Configuration dictionary.
        name: Run name (logged as parameter).
        notes: Run notes (logged as text).
        tags: Run tags.
        dir: Directory for logs (ignored).
        resume: Whether to resume (ignored).
        reinit: Whether to reinit (ignored).
        mode: Mode (ignored).
        **kwargs: Additional arguments (ignored).

    Returns:
        WandbWriter instance.
    """
    writer = _get_or_create_writer()

    # Log run metadata as parameters
    if project:
        writer._log_internal("wandb/project", project, AnalyticsDataType.PARAMETER)
    if entity:
        writer._log_internal("wandb/entity", entity, AnalyticsDataType.PARAMETER)
    if name:
        writer._log_internal("wandb/run_name", name, AnalyticsDataType.PARAMETER)
    if notes:
        writer._log_internal("wandb/notes", notes, AnalyticsDataType.TEXT)
    if tags:
        writer._log_internal("wandb/tags", list(tags), AnalyticsDataType.TAG)

    # Log configuration
    if config:
        for key, value in config.items():
            writer._log_internal(f"config/{key}", value, AnalyticsDataType.PARAMETER)

    return writer


def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None,
):
    """Log metrics (module-level function)."""
    _get_or_create_writer().log(data, step, commit, sync)


def define_metric(
    name: str,
    step_metric: Optional[str] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
):
    """Define a custom metric (module-level function)."""
    _get_or_create_writer().define_metric(name, step_metric, step_sync, hidden, summary, goal, overwrite)


def finish(exit_code: Optional[int] = None, quiet: Optional[bool] = None):
    """Finish the wandb run (no-op in subprocess mode)."""
    global _wandb_writer
    if _wandb_writer:
        _wandb_writer.close()
        _wandb_writer = None
