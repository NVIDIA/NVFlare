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

"""Collab tracking module for experiment metrics.

This module provides tracking writers that mimic the APIs of popular
experiment tracking libraries (TensorBoard, MLflow, W&B). The writers
work in BOTH in-process and subprocess execution modes:

- In-process mode: Fires events directly to event_manager
- Subprocess mode: Sends metrics via CellNet to parent CollabExecutor

Users just change the import statement - the execution mode is
automatically detected.

Example - TensorBoard:
    # Original:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_scalar("loss", 0.5, global_step=100)

    # Collab (same API, works in both modes):
    from nvflare.collab.tracking import SummaryWriter
    writer = SummaryWriter()
    writer.add_scalar("loss", 0.5, global_step=100)

Example - MLflow:
    # Original:
    import mlflow
    mlflow.log_metric("loss", 0.5, step=100)

    # Collab (same API):
    from nvflare.collab.tracking import mlflow
    mlflow.log_metric("loss", 0.5, step=100)

Example - W&B:
    # Original:
    import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5})

    # Collab (same API):
    from nvflare.collab.tracking import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5})

The public surface is the compat writers above; the mode-detection and
CellNet relay machinery lives in the private ``_transport`` subpackage.
Exports are resolved lazily (PEP 562) so importing this package stays
cheap and does not pull in the transport stack.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._transport.auto import AutoWriter
    from ._transport.auto import get_writer as get_auto_writer
    from ._transport.base import BaseWriter
    from ._transport.in_process import InProcessWriter, get_inprocess_writer, set_inprocess_writer
    from ._transport.relay import MetricsRelay, setup_metrics_handler
    from ._transport.subprocess import SubprocessWriter, get_writer, set_writer
    from .mlflow import MLflowWriter
    from .tensorboard import SummaryWriter, TensorBoardWriter
    from .wandb import WandbWriter

__all__ = [
    "AutoWriter",
    "BaseWriter",
    "InProcessWriter",
    "MLflowWriter",
    "MetricsRelay",
    "SubprocessWriter",
    "SummaryWriter",
    "TensorBoardWriter",
    "WandbWriter",
    "get_auto_writer",
    "get_inprocess_writer",
    "get_writer",
    "mlflow",
    "set_inprocess_writer",
    "set_writer",
    "setup_metrics_handler",
    "wandb",
]

# name -> (module, attribute); attribute None means the submodule itself
# (the mlflow/wandb compat modules are used as drop-in module replacements).
_EXPORTS = {
    "mlflow": (".mlflow", None),
    "wandb": (".wandb", None),
    "AutoWriter": ("._transport.auto", "AutoWriter"),
    "get_auto_writer": ("._transport.auto", "get_writer"),
    "BaseWriter": ("._transport.base", "BaseWriter"),
    "InProcessWriter": ("._transport.in_process", "InProcessWriter"),
    "get_inprocess_writer": ("._transport.in_process", "get_inprocess_writer"),
    "set_inprocess_writer": ("._transport.in_process", "set_inprocess_writer"),
    "MetricsRelay": ("._transport.relay", "MetricsRelay"),
    "setup_metrics_handler": ("._transport.relay", "setup_metrics_handler"),
    "MLflowWriter": (".mlflow", "MLflowWriter"),
    "SubprocessWriter": ("._transport.subprocess", "SubprocessWriter"),
    "get_writer": ("._transport.subprocess", "get_writer"),
    "set_writer": ("._transport.subprocess", "set_writer"),
    "SummaryWriter": (".tensorboard", "SummaryWriter"),
    "TensorBoardWriter": (".tensorboard", "TensorBoardWriter"),
    "WandbWriter": (".wandb", "WandbWriter"),
}


def __getattr__(name):
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod_path, attr = export
    module = importlib.import_module(mod_path, __package__)
    return module if attr is None else getattr(module, attr)


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
